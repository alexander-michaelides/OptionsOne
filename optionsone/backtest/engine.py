"""Chronological replay of news events and completed minute bars."""

from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from heapq import heappop, heappush
from math import isfinite
from typing import Final

from optionsone.backtest.portfolio import Portfolio
from optionsone.backtest.report import EXCHANGE_TZ, BacktestReport, EquityPoint, build_report
from optionsone.backtest.strategy import Strategy
from optionsone.core.clock import SimClock
from optionsone.core.pit import LookaheadError, PointInTimeBars
from optionsone.core.types import Bar, OrderIntent, Position
from optionsone.execution.costs import DEFAULT_COSTS, CostModel
from optionsone.execution.order_manager import OrderManager
from optionsone.execution.sim_broker import LiquiditySource, SimBroker, SimExecution
from optionsone.ingest.calendar import session_close as exchange_session_close
from optionsone.risk.kill_switch import KillSwitch
from optionsone.risk.limits import RiskLimits, pre_trade_check
from optionsone.risk.sizing import SizingConfig, realized_bar_vol, size_shares

_DEFAULT_SIZING: Final = SizingConfig()
_DEFAULT_LIMITS: Final = RiskLimits()


class InvalidReplayInput(ValueError):
    """Replay data violates a timeline or account precondition."""


@dataclass(frozen=True, slots=True)
class RejectedOrder:
    order: OrderIntent
    reason: str


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity_curve: tuple[EquityPoint, ...]
    executions: tuple[SimExecution, ...]
    rejected_orders: tuple[RejectedOrder, ...]
    report: BacktestReport
    open_positions: tuple[Position, ...] = ()


async def run_backtest[Payload](
    strategy: Strategy[Payload],
    bars: Iterable[Bar],
    *,
    initial_cash: float = 100_000.0,
    sizing: SizingConfig = _DEFAULT_SIZING,
    limits: RiskLimits = _DEFAULT_LIMITS,
    costs: CostModel = DEFAULT_COSTS,
    latency: timedelta = timedelta(),
    liquidity: LiquiditySource | None = None,
    kill_switch: KillSwitch | None = None,
    session_close: Callable[[date], datetime | None] = exchange_session_close,
    flatten_before: timedelta = timedelta(minutes=5),
) -> BacktestResult:
    """Replay regular-session bars and submit exits from the calendar-defined cutoff."""
    if not isfinite(initial_cash) or initial_cash <= 0:
        raise InvalidReplayInput("initial_cash must be finite and positive")
    if flatten_before < timedelta():
        raise InvalidReplayInput("flatten_before must be nonnegative")
    closes: dict[datetime, list[Bar]] = defaultdict(list)
    for bar in sorted(bars, key=lambda item: (item.end, item.symbol)):
        if closes[bar.end] and closes[bar.end][-1].symbol == bar.symbol:
            raise InvalidReplayInput(f"duplicate bar for {bar.symbol} at {bar.start}")
        closes[bar.end].append(bar)

    events = iter(strategy.events())
    event = next(events, None)
    timeline = sorted(closes)
    scheduled = set(timeline)
    start = (
        min(timeline[0], event[0])
        if timeline and event
        else (timeline[0] if timeline else event[0] if event else None)
    )
    portfolio = Portfolio(initial_cash)
    curve: list[EquityPoint] = []
    executions: list[SimExecution] = []
    rejected: list[RejectedOrder] = []
    if start is None:
        return BacktestResult((), (), (), build_report(initial_cash, (), (), ()))

    clock = SimClock(start)
    pit_bars = PointInTimeBars((bar for group in closes.values() for bar in group), clock)
    broker = SimBroker(pit_bars, clock, costs=costs, latency=latency, liquidity=liquidity)
    manager = OrderManager(broker)
    marks: dict[str, float] = {}
    orders: deque[datetime] = deque()

    while timeline or event is not None:
        if timeline and (event is None or timeline[0] <= event[0]):
            now = heappop(timeline)
            scheduled.remove(now)
        elif event is not None:
            now = event[0]
        else:
            break
        clock.advance_to(now)
        session = now.astimezone(EXCHANGE_TZ).date()
        close = session_close(session)
        flattening = close is not None and now >= close - flatten_before
        if kill_switch is not None:
            kill_switch.observe(session, portfolio.equity)
        payloads: list[Payload] = []
        while event is not None and event[0] == now:
            payloads.append(event[1])
            event = next(events, None)
            if event is not None and event[0] < now:
                raise InvalidReplayInput("strategy events must be in knowledge-time order")

        closing_bars = closes.get(now, ())
        for fill in await manager.collect_fills():
            execution = broker.execution(fill.client_order_id)
            executions.append(execution)
            portfolio.apply(execution)
        for bar in closing_bars:
            marks[bar.symbol] = bar.close
            portfolio.mark(bar.symbol, bar.close)

        if kill_switch is not None:
            was_latched = kill_switch.flatten_requested
            kill_switch.observe(session, portfolio.equity)
            if kill_switch.flatten_requested and not was_latched:
                await manager.cancel_working()
        halted = kill_switch is not None and kill_switch.flatten_requested
        if halted or flattening:
            due = tuple(
                position.symbol
                for position in portfolio.positions
                if not manager.has_working_order(position.symbol)
            )
            exit_reason = "kill_switch" if halted else "session_close"
        else:
            due = manager.due_symbols(now)
            exit_reason = "horizon"
        for symbol in due:
            if await manager.submit(manager.exit_order(symbol, now, exit_reason)):
                orders.append(now)

        while orders and orders[0] <= now - timedelta(minutes=1):
            orders.popleft()
        if closing_bars or payloads:
            signals = strategy.on_event(now, pit_bars, portfolio.snapshot(tuple(payloads)))
            for signal in signals:
                if signal.as_of > now:
                    raise LookaheadError(f"signal as-of {signal.as_of} is after decision {now}")
                deadline = signal.as_of + signal.horizon
                if close is None or flattening or now + latency >= close or deadline <= now:
                    continue
                if manager.has_working_order(signal.ticker) or any(
                    position.symbol == signal.ticker for position in portfolio.positions
                ):
                    continue
                visible = pit_bars.bars_asof(signal.ticker)
                vol = realized_bar_vol(visible, sizing.vol_window)
                if vol is None or visible[-1].end.astimezone(EXCHANGE_TZ).date() != session:
                    continue
                qty = size_shares(signal, portfolio.equity, visible[-1].close, vol, sizing)
                if not qty:
                    continue
                order = manager.entry_order(signal, qty, now)
                if kill_switch is not None and not kill_switch.allows(order, 0):
                    rejected.append(RejectedOrder(order, kill_switch.reason or "kill_switch"))
                    continue
                decision = pre_trade_check(
                    order,
                    limits,
                    equity=portfolio.equity,
                    quantities=manager.projected_quantities(),
                    prices=marks,
                    submitted_at=orders,
                )
                if not decision.allowed:
                    rejected.append(RejectedOrder(order, decision.reason))
                    continue
                if await manager.submit(order, deadline=deadline):
                    orders.append(now)
                    if deadline < close and deadline not in scheduled:
                        heappush(timeline, deadline)
                        scheduled.add(deadline)
        if closing_bars:
            curve.append(EquityPoint(now, portfolio.equity))

    report = build_report(initial_cash, curve, portfolio.trades, executions)
    return BacktestResult(
        tuple(curve), tuple(executions), tuple(rejected), report, portfolio.positions
    )
