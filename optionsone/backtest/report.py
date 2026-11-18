"""Performance metrics from a completed replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import sqrt
from statistics import fmean, stdev
from typing import Final
from zoneinfo import ZoneInfo

from optionsone.backtest.portfolio import Trade
from optionsone.execution.costs import ExecutionCosts
from optionsone.execution.sim_broker import SimExecution

EXCHANGE_TZ: Final = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    time: datetime
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestReport:
    trades: tuple[Trade, ...]
    gross_return: float
    net_return: float
    sharpe_per_trade: float
    sharpe_per_day: float
    max_drawdown: float
    turnover: float
    costs: ExecutionCosts
    cost_sensitivity: tuple[tuple[int, float], ...]


def _sharpe(returns: Sequence[float], scale: float = 1.0) -> float:
    deviation = stdev(returns) if len(returns) > 1 else 0.0
    return sqrt(scale) * fmean(returns) / deviation if deviation else 0.0


def build_report(
    initial_cash: float,
    curve: Sequence[EquityPoint],
    trades: tuple[Trade, ...],
    executions: Sequence[SimExecution],
) -> BacktestReport:
    """Compute 0x/1x/2x cost sensitivity holding orders and reference prices fixed."""
    peak = initial_cash
    drawdown = 0.0
    daily_end: dict[date, float] = {}
    for point in curve:
        peak = max(peak, point.equity)
        drawdown = max(drawdown, 1 - point.equity / peak)
        daily_end[point.time.astimezone(EXCHANGE_TZ).date()] = point.equity
    daily_returns: list[float] = []
    prior = initial_cash
    for equity in daily_end.values():
        if prior > 0:
            daily_returns.append(equity / prior - 1)
        prior = equity
    costs = ExecutionCosts(
        sum(execution.costs.commission for execution in executions),
        sum(execution.costs.spread for execution in executions),
        sum(execution.costs.impact for execution in executions),
    )
    net_pnl = (curve[-1].equity if curve else initial_cash) - initial_cash
    gross_pnl = net_pnl + costs.total
    return BacktestReport(
        trades=trades,
        gross_return=gross_pnl / initial_cash,
        net_return=net_pnl / initial_cash,
        sharpe_per_trade=_sharpe([t.net_pnl / t.entry_notional for t in trades]),
        sharpe_per_day=_sharpe(daily_returns, 252),
        max_drawdown=drawdown,
        turnover=sum(e.fill.qty * e.reference_price for e in executions) / initial_cash,
        costs=costs,
        cost_sensitivity=tuple(
            (multiple, (gross_pnl - multiple * costs.total) / initial_cash)
            for multiple in (0, 1, 2)
        ),
    )
