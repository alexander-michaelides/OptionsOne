"""Historical fills through the frozen point-in-time bar interface."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from optionsone.core.clock import Clock
from optionsone.core.pit import LookaheadError, PointInTimeBars
from optionsone.core.ports import Broker
from optionsone.core.types import Fill, OrderIntent, Position
from optionsone.execution.costs import DEFAULT_COSTS, CostModel, ExecutionCosts, Liquidity
from optionsone.execution.order_manager import position_after_fill

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
                self._execute(order, bar.open, bar.start)
                del self._pending[order.client_order_id]

    def execution(self, client_id: str) -> SimExecution:
        return self._executions[client_id]

    def _execute(self, order: OrderIntent, reference: float, when: datetime) -> None:
        liquidity = self._liquidity(order.symbol, order.created_at) if self._liquidity else None
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

    async def cancel(self, client_order_ids: Sequence[str]) -> None:
        for client_order_id in client_order_ids:
            self._pending.pop(client_order_id, None)


class NegativeLatencyError(ValueError):
    def __init__(self, latency: timedelta) -> None:
        super().__init__(f"latency cannot be negative: {latency}")
