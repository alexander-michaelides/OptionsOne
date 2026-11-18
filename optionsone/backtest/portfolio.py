"""Cash and signed inventory ledger for historical fills."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from optionsone.core.types import OrderSide, Position
from optionsone.execution.order_manager import position_after_fill
from optionsone.execution.sim_broker import SimExecution


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    quantity: int
    entry_at: datetime
    exit_at: datetime
    entry_notional: float
    gross_pnl: float
    net_pnl: float


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot[Payload]:
    cash: float
    equity: float
    positions: tuple[Position, ...]
    realized_pnl: float
    unrealized_pnl: float
    events: tuple[Payload, ...]


@dataclass(frozen=True, slots=True)
class _Lot:
    execution: SimExecution
    quantity: int


class Portfolio:
    """Apply fills to cash and match closing shares against FIFO opening lots."""

    def __init__(self, initial_cash: float) -> None:
        self.cash = initial_cash
        self.realized_pnl = 0.0
        self._positions: dict[str, Position] = {}
        self._lots: dict[str, deque[_Lot]] = {}
        self._marks: dict[str, float] = {}
        self._trades: list[Trade] = []

    @property
    def positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    @property
    def trades(self) -> tuple[Trade, ...]:
        return tuple(self._trades)

    @property
    def equity(self) -> float:
        return self.cash + sum(p.qty * self._marks[p.symbol] for p in self.positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            lot.quantity
            * (
                (self._marks[symbol] - lot.execution.fill.price)
                * (1 if lot.execution.fill.side is OrderSide.BUY else -1)
                - lot.execution.fill.fee / lot.execution.fill.qty
            )
            for symbol, lots in self._lots.items()
            for lot in lots
        )

    def mark(self, symbol: str, price: float) -> None:
        self._marks[symbol] = price

    def snapshot[Payload](self, events: tuple[Payload, ...]) -> PortfolioSnapshot[Payload]:
        return PortfolioSnapshot(
            self.cash, self.equity, self.positions, self.realized_pnl, self.unrealized_pnl, events
        )

    def apply(self, execution: SimExecution) -> None:
        fill = execution.fill
        sign = 1 if fill.side is OrderSide.BUY else -1
        self.cash -= sign * fill.qty * fill.price + fill.fee
        self._marks.setdefault(fill.symbol, execution.reference_price)
        lots = self._lots.setdefault(fill.symbol, deque())
        remaining = fill.qty
        while remaining and lots and lots[0].execution.fill.side is not fill.side:
            lot = lots.popleft()
            entry = lot.execution
            quantity = min(remaining, lot.quantity)
            gross = sign * (entry.reference_price - execution.reference_price) * quantity
            net = sign * (entry.fill.price - fill.price) * quantity - quantity * (
                entry.fill.fee / entry.fill.qty + fill.fee / fill.qty
            )
            self._trades.append(
                Trade(
                    fill.symbol,
                    quantity,
                    entry.fill.filled_at,
                    fill.filled_at,
                    quantity * entry.reference_price,
                    gross,
                    net,
                )
            )
            self.realized_pnl += net
            if quantity < lot.quantity:
                lots.appendleft(_Lot(entry, lot.quantity - quantity))
            remaining -= quantity
        if remaining:
            lots.append(_Lot(execution, remaining))
        position = position_after_fill(self._positions.get(fill.symbol), fill)
        if position is None:
            self._positions.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = position
