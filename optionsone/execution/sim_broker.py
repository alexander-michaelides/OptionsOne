"""Historical fills through the frozen point-in-time bar interface."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from optionsone.core.clock import Clock
from optionsone.core.pit import LookaheadError, PointInTimeBars
from optionsone.core.ports import Broker
from optionsone.core.types import Bar, Fill, OrderIntent, Position
from optionsone.execution.costs import DEFAULT_COSTS, CostModel, ExecutionCosts, Liquidity
from optionsone.execution.order_manager import position_after_fill
from optionsone.risk.limits import signed_qty

type LiquiditySource = Callable[[str, datetime], Liquidity | None]


@dataclass(frozen=True, slots=True)
class SimExecution:
    fill: Fill
    reference_price: float
    costs: ExecutionCosts


class SimBroker(Broker):
    def __init__(
        self,
        bars: PointInTimeBars,
        clock: Clock,
        *,
        costs: CostModel = DEFAULT_COSTS,
        latency: timedelta = timedelta(),
        liquidity: LiquiditySource | None = None,
    ) -> None:
        if latency < timedelta():
            raise NegativeLatencyError(latency)
        self._bars = bars
        self._clock = clock
        self._costs = costs
        self._latency = latency
        self._liquidity = liquidity
        self._seen: set[str] = set()
        self._pending: dict[str, OrderIntent] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, Position] = {}
        self._executions: dict[str, SimExecution] = {}

    def eligible_at(self, order: OrderIntent) -> datetime:
        return order.created_at + self._latency

    async def submit(self, order: OrderIntent) -> None:
        if order.client_order_id in self._seen:
            return
        if order.created_at < self._clock.now():
            raise LookaheadError("cannot submit a simulated order in the past")
        self._seen.add(order.client_order_id)
        self._pending[order.client_order_id] = order

    def _release_completed_opens(self) -> None:
        latest = {
            symbol: self._bars.last_bar(symbol)
            for symbol in {order.symbol for order in self._pending.values()}
        }
        for order in tuple(self._pending.values()):
            symbol = order.symbol
            eligible_at = self.eligible_at(order)
            visible = latest[symbol]
            if visible is None or visible.start < eligible_at:
                continue
            bar = self._bars.next_bar_after(symbol, eligible_at)
            if bar is not None:
                if bar.end > self._clock.now() or bar.start < eligible_at:
                    raise LookaheadError(f"ineligible fill bar for {symbol} at {bar.start}")
                self._execute(order, bar.open, bar.start, order.created_at)
                del self._pending[order.client_order_id]

    def settle_session_close(self, order: OrderIntent, bar: Bar) -> None:
        """Settle the replay's standing liquidation instruction at the final close."""
        if bar.end != self._clock.now() or self._bars.last_bar(order.symbol) != bar:
            raise LookaheadError(f"unavailable closing bar for {order.symbol} at {bar.start}")
        position = self._positions.get(order.symbol)
        if position is None or position.qty != -signed_qty(order):
            raise InvalidSessionLiquidation(order.symbol)
        self._execute(order, bar.close, bar.end, bar.start)

    def execution(self, client_id: str) -> SimExecution:
        return self._executions[client_id]

    def _execute(
        self, order: OrderIntent, reference: float, when: datetime, liquidity_asof: datetime
    ) -> None:
        liquidity = self._liquidity(order.symbol, liquidity_asof) if self._liquidity else None
        price, costs = self._costs.quote(reference, order.qty, order.side, liquidity)
        fill = Fill(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            fee=costs.commission,
            filled_at=when,
        )
        position = position_after_fill(self._positions.get(fill.symbol), fill)
        if position is None:
            self._positions.pop(fill.symbol)
        else:
            self._positions[fill.symbol] = position
        self._fills.append(fill)
        self._executions[order.client_order_id] = SimExecution(fill, reference, costs)

    async def fills(self) -> Sequence[Fill]:
        self._release_completed_opens()
        fills, self._fills = self._fills, []
        return sorted(fills, key=lambda fill: fill.filled_at)

    async def positions(self) -> Sequence[Position]:
        return tuple(self._positions.values())

    def cancel_symbol(self, symbol: str) -> None:
        self._pending = {
            key: order for key, order in self._pending.items() if order.symbol != symbol
        }

    async def cancel_all(self) -> None:
        self._pending.clear()


class NegativeLatencyError(ValueError):
    def __init__(self, latency: timedelta) -> None:
        super().__init__(f"latency cannot be negative: {latency}")


class InvalidSessionLiquidation(ValueError):
    def __init__(self, symbol: str) -> None:
        super().__init__(f"closing auction must flatten the entire position in {symbol}")
