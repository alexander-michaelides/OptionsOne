from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from optionsone.backtest.engine import run_backtest
from optionsone.backtest.portfolio import PortfolioSnapshot
from optionsone.core.pit import LookaheadError, PointInTimeBars
from optionsone.core.types import Bar, Side, Signal
from optionsone.execution.costs import CostModel, Liquidity
from optionsone.risk.kill_switch import KillSwitch, KillSwitchConfig

START = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def bar(minute: int, opening: float, close: float) -> Bar:
    return Bar(
        symbol="A",
        start=START + minute * MINUTE,
        open=opening,
        high=max(opening, close),
        low=min(opening, close),
        close=close,
        volume=1000,
    )


def signal(when: datetime, horizon: timedelta, article_id: int = 1) -> Signal:
    return Signal(
        article_id=article_id,
        ticker="A",
        as_of=when,
        side=Side.LONG,
        strength=1,
        horizon=horizon,
        model_id="test",
    )


class NewsStrategy:
    def __init__(self, *signals: Signal) -> None:
        self.signals = signals
        self.observed: dict[datetime, PortfolioSnapshot[Signal]] = {}

    def events(self) -> Iterator[tuple[datetime, Signal]]:
        return ((item.as_of, item) for item in self.signals)

    def on_event(
        self, now: datetime, pit_bars: PointInTimeBars, portfolio: PortfolioSnapshot[Signal]
    ) -> Sequence[Signal]:
        assert all(item.end <= now for item in pit_bars.bars_asof("A"))
        self.observed[now] = portfolio
        return portfolio.events


async def test_golden_news_trade_matches_hand_calculated_net_pnl() -> None:
    bars = [
        bar(0, 100, 100),
        bar(1, 100, 100),
        bar(2, 101, 102),
        bar(3, 104, 105),
        bar(4, 106, 107),
    ]
    decision = START + 2 * MINUTE
    strategy = NewsStrategy(signal(decision, MINUTE))
    result = await run_backtest(
        strategy,
        bars,
        initial_cash=10_000,
        costs=CostModel(commission_per_share=0.01, half_spread_bps=10),
    )

    # Buy 10 at 101 + .101, sell 10 at 104 - .104, pay .01/share twice.
    # Net = 10 * (103.896 - 101.101) - .20 = 27.75.
    (trade,) = result.report.trades
    assert trade.net_pnl == pytest.approx(27.75, rel=0, abs=1e-9)
    assert result.equity_curve[-1].equity == pytest.approx(10_027.75, rel=0, abs=1e-9)
    assert trade.entry_at == decision
    assert trade.exit_at == decision + MINUTE
    assert strategy.observed[decision].positions == ()
    assert strategy.observed[decision + MINUTE].unrealized_pnl == pytest.approx(8.89)
    assert result.report.turnover == pytest.approx(0.205)
    assert dict(result.report.cost_sensitivity) == pytest.approx(
        {0: 0.003, 1: 0.002775, 2: 0.00255}
    )


async def test_latency_uses_first_eligible_open_and_session_liquidates_at_close() -> None:
    bars = [
        bar(0, 100, 100),
        bar(1, 100, 100),
        bar(2, 105, 106),
        bar(3, 110, 111),
        bar(4, 120, 121),
    ]
    decision = START + 2 * MINUTE
    strategy = NewsStrategy(signal(decision, 20 * MINUTE))
    result = await run_backtest(
        strategy,
        bars,
        initial_cash=10_000,
        latency=timedelta(seconds=1),
        costs=CostModel(commission_per_share=0, half_spread_bps=0, eta=0.2),
        liquidity=lambda _symbol, _as_of: Liquidity(sigma=0.02, adv=1000),
    )

    entry, exit_ = result.executions
    assert (entry.fill.filled_at, entry.reference_price) == (bars[3].start, 110)
    assert (exit_.fill.filled_at, exit_.reference_price) == (bars[4].end, 121)
    assert result.report.costs.impact == pytest.approx(0.924)
    assert strategy.observed[bars[4].end].positions == ()


async def test_engine_propagates_lookahead_when_forced_to_read_incomplete_fill_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PointInTimeBars.next_bar_after

    def force_later_bar(self: PointInTimeBars, symbol: str, when: datetime) -> Bar | None:
        return original(self, symbol, when + MINUTE)

    monkeypatch.setattr(PointInTimeBars, "next_bar_after", force_later_bar)
    strategy = NewsStrategy(signal(START + 2 * MINUTE, 10 * MINUTE))
    bars = [bar(i, 100, 100) for i in range(5)]

    with pytest.raises(LookaheadError, match="not complete"):
        await run_backtest(strategy, bars)


async def test_drawdown_breach_exits_at_next_open_and_blocks_reentry() -> None:
    strategy = NewsStrategy(
        signal(START + 2 * MINUTE, 10 * MINUTE),
        signal(START + 4 * MINUTE, 10 * MINUTE, article_id=2),
    )
    bars = [
        bar(0, 100, 100),
        bar(1, 100, 100),
        bar(2, 100, 75),
        bar(3, 72, 73),
        bar(4, 74, 75),
    ]
    switch = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02, reset_code=SecretStr("code")))

    result = await run_backtest(strategy, bars, initial_cash=10_000, kill_switch=switch)

    (trade,) = result.report.trades
    assert trade.exit_at == bars[3].start
    assert result.executions[-1].reference_price == 72
    (rejected,) = result.rejected_orders
    assert rejected.reason == "daily_loss"
