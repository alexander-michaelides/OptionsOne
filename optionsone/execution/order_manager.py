"""Idempotent order submission, signed inventory and signal deadlines."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from optionsone.core.ports import Broker
from optionsone.core.types import Fill, OrderIntent, OrderSide, Position, Side, Signal
from optionsone.risk.limits import signed_qty


def client_order_id(signal: Signal) -> str:
    identity = (signal.article_id, signal.ticker, signal.side.value, signal.as_of.isoformat())
    return sha256(repr(identity).encode()).hexdigest()[:48]


def position_after_fill(previous: Position | None, fill: Fill) -> Position | None:
    old_qty = previous.qty if previous else 0
    change = fill.qty if fill.side is OrderSide.BUY else -fill.qty
    qty = old_qty + change
    if not qty:
        return None
    price = fill.price
    if previous and old_qty * qty > 0:
        price = previous.avg_price
        if old_qty * change > 0:
            price = (price * abs(old_qty) + fill.price * fill.qty) / abs(qty)
    return Position(symbol=fill.symbol, qty=qty, avg_price=price)


@dataclass(frozen=True, slots=True)
class _WorkingOrder:
    order: OrderIntent
    remaining: int
    deadline: datetime | None


class OrderManager:
    """Manages one signal holding per symbol, with exits measured from signal as-of."""

    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._seen: set[str] = set()
        self._working: dict[str, _WorkingOrder] = {}
        self._positions: dict[str, Position] = {}
        self._deadlines: dict[str, datetime] = {}

    @staticmethod
    def entry_order(signal: Signal, qty: int, now: datetime) -> OrderIntent:
        return OrderIntent(
            client_order_id=client_order_id(signal),
            symbol=signal.ticker,
            side=OrderSide.BUY if signal.side is Side.LONG else OrderSide.SELL,
            qty=qty,
            created_at=now,
            reason="signal",
        )

    def exit_order(self, symbol: str, now: datetime, reason: str) -> OrderIntent:
        position = self._positions[symbol]
        identity = (symbol, now.isoformat(), reason, position.qty)
        digest = sha256(repr(identity).encode()).hexdigest()
        return OrderIntent(
            client_order_id=f"exit-{digest[:43]}",
            symbol=symbol,
            side=OrderSide.SELL if position.qty > 0 else OrderSide.BUY,
            qty=abs(position.qty),
            created_at=now,
            reason=reason,
        )

    def has_working_order(self, symbol: str) -> bool:
        return any(work.order.symbol == symbol for work in self._working.values())

    def projected_quantities(self) -> dict[str, int]:
        quantities = {symbol: position.qty for symbol, position in self._positions.items()}
        for work in self._working.values():
            order = work.order
            quantities[order.symbol] = (
                quantities.get(order.symbol, 0) + signed_qty(order) // order.qty * work.remaining
            )
        return quantities

    async def submit(self, order: OrderIntent, *, deadline: datetime | None = None) -> bool:
        key = order.client_order_id
        if key in self._seen:
            return False
        self._seen.add(key)
        try:
            await self._broker.submit(order)
            self._working[key] = _WorkingOrder(order, order.qty, deadline)
        finally:
            if key not in self._working:
                self._seen.remove(key)
        return True

    def record_fill(self, fill: Fill) -> None:
        working = self._working.pop(fill.client_order_id, None)
        if working is not None:
            if working.deadline is not None:
                self._deadlines[fill.symbol] = working.deadline
            if working.remaining > fill.qty:
                self._working[fill.client_order_id] = _WorkingOrder(
                    working.order, working.remaining - fill.qty, working.deadline
                )
        position = position_after_fill(self._positions.get(fill.symbol), fill)
        if position is None:
            self._positions.pop(fill.symbol, None)
            self._deadlines.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = position

    async def collect_fills(self) -> Sequence[Fill]:
        fills = await self._broker.fills()
        for fill in fills:
            self.record_fill(fill)
        return fills

    def due_symbols(self, now: datetime) -> tuple[str, ...]:
        return tuple(
            symbol
            for symbol, deadline in sorted(self._deadlines.items())
            if deadline <= now and not self.has_working_order(symbol)
        )

    def forget_working(self, symbol: str) -> None:
        """Release reservations after broker-side cancellation."""
        self._working = {
            key: work for key, work in self._working.items() if work.order.symbol != symbol
        }
