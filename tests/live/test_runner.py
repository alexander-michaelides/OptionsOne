from datetime import timedelta
from pathlib import Path

import pytest

from optionsone.app.strategy import NewsStrategy
from optionsone.backtest.portfolio import Portfolio
from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Fill, OrderSide, Position
from optionsone.ingest.calendar import is_regular_session
from optionsone.live.bars import LiveBarCache
from tests.live.scenario import CUTOFF, HORIZON, MINUTE, MODEL, START, Scenario


async def test_positive_article_submits_one_paper_entry(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        await scenario.news.deliver(scenario.article())
    (order,) = scenario.broker.orders
    assert (order.symbol, order.side, order.qty, order.reason) == (
        "ACME",
        OrderSide.BUY,
        66,
        "signal",
    )
    assert not scenario.broker.working
    assert [(record.event, record.reason) for record in scenario.records()] == [
        ("entry", "signal"),
        ("cancel", "shutdown"),
    ]


async def test_replay_is_deduplicated_even_after_exit(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        original = scenario.article()
        await scenario.news.deliver(original)
        scenario.broker.fill(scenario.broker.orders[0])
        await scenario.tick(START + HORIZON)
        scenario.broker.fill(scenario.broker.orders[1])
        await scenario.tick(START + HORIZON + MINUTE)
        await scenario.news.deliver(original)
    assert [order.reason for order in scenario.broker.orders] == ["signal", "horizon"]
    assert scenario.classifier.calls == [(1, "ACME")]
    assert [r.article_id for r in scenario.records() if r.reason == "duplicate_article"] == [1]


async def test_horizon_exit_waits_for_deadline(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        await scenario.news.deliver(scenario.article())
        scenario.broker.fill(scenario.broker.orders[0])
        await scenario.tick(START + HORIZON - timedelta(microseconds=1))
        assert len(scenario.broker.orders) == 1
        await scenario.tick(START + HORIZON)
        await scenario.tick(START + HORIZON + MINUTE)
    entry, exit_order = scenario.broker.orders
    assert (exit_order.reason, exit_order.created_at) == ("horizon", START + HORIZON)
    assert (exit_order.side, exit_order.qty) == (OrderSide.SELL, entry.qty)


async def test_cutoff_blocks_news_before_ticker_and_flattens_on_tick(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        await scenario.news.deliver(scenario.article())
        scenario.broker.fill(scenario.broker.orders[0])
        scenario.clock.advance_to(CUTOFF)
        await scenario.news.deliver(scenario.article(2, "OTHER"))
        assert len(scenario.broker.orders) == 1
        await scenario.tick(CUTOFF)
    assert [order.reason for order in scenario.broker.orders] == ["signal", "session_close"]
    assert [r.reason for r in scenario.records() if r.article_id == 2] == ["session_cutoff"]


async def test_equity_loss_latches_and_journals_later_entry_block(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        scenario.broker.account_value = 90_000
        await scenario.tick(START + MINUTE)
        scenario.broker.account_value = 100_000
        await scenario.news.deliver(scenario.article())
    assert not scenario.broker.orders
    assert [(r.event, r.reason) for r in scenario.records() if r.article_id == 1] == [
        ("skip", "daily_loss"),
    ]


async def test_symbol_outside_universe_is_journaled_without_classifying(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        await scenario.news.deliver(scenario.article(symbol="UNKNOWN"))
    assert not scenario.classifier.calls
    assert not scenario.broker.orders
    assert [(r.symbol, r.reason) for r in scenario.records() if r.article_id == 1] == [
        ("UNKNOWN", "outside_universe"),
    ]


async def test_classifier_exception_is_journaled_and_next_article_trades(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    async with scenario.running():
        await scenario.news.deliver(scenario.article(headline="failure"))
        await scenario.news.deliver(scenario.article(2))
    assert len(scenario.broker.orders) == 1
    assert [(r.article_id, r.event, r.reason) for r in scenario.records() if r.article_id] == [
        (1, "error", "classifier_error"),
        (2, "entry", "signal"),
    ]


async def test_replay_strategy_and_paper_runner_emit_same_signal(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    article = scenario.article()
    classification = await scenario.classifier.classify(article, "ACME")
    async with scenario.running():
        await scenario.news.deliver(article)
        replay = NewsStrategy([classification], MODEL, is_regular_session)
        expected = replay.on_event(
            START,
            PointInTimeBars(scenario.bars.last_bars, scenario.clock),
            Portfolio(100_000).snapshot((classification,)),
        )
    (live,) = [r.signal for r in scenario.records() if r.event == "entry"]
    assert live is not None
    assert list(expected) == [live]


async def test_bar_cache_fetches_spy_and_exposes_only_completed_bars(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    cache = LiveBarCache(scenario.bars, scenario.clock)
    scenario.clock.advance_to(START + timedelta(seconds=30))
    first = await cache.refresh({"ACME"})
    await cache.refresh({"SPY", "ACME"})
    (request,) = scenario.bars.requests
    assert request[0] == ("ACME", "SPY")
    assert request[1] == START - 45 * MINUTE
    assert [b.end for b in first.bars_asof("ACME")][-1] == START
    scenario.clock.advance_to(START + MINUTE)
    second = await cache.refresh({"ACME"})
    assert [b.end for b in second.bars_asof("ACME")][-1] == START + MINUTE


@pytest.mark.parametrize("exit_reason", ["horizon", "session_close", "kill_switch", "shutdown"])
async def test_foreign_positions_never_exit_or_use_position_slots(
    tmp_path: Path,
    exit_reason: str,
) -> None:
    scenario = Scenario(tmp_path / "events.jsonl", flatten_on_stop=exit_reason == "shutdown")
    foreign = Position(symbol="OTHER", qty=1, avg_price=150)
    scenario.broker.inventory[foreign.symbol] = foreign
    async with scenario.running():
        await scenario.news.deliver(scenario.article(symbol=foreign.symbol))
        await scenario.news.deliver(scenario.article(2))
        (entry,) = scenario.broker.orders
        assert (entry.symbol, entry.reason) == ("ACME", "signal")
        scenario.broker.fill(entry)
        scenario.broker.pending_fills.append(
            Fill(
                client_order_id="manual-order",
                symbol=foreign.symbol,
                side=OrderSide.BUY,
                qty=1,
                price=150,
                fee=0,
                filled_at=START,
            )
        )
        if exit_reason == "kill_switch":
            scenario.broker.account_value = 90_000
        if exit_reason != "shutdown":
            await scenario.tick(CUTOFF if exit_reason == "session_close" else START + HORIZON)
            scenario.broker.fill(scenario.broker.orders[1])
    assert [(order.symbol, order.reason) for order in scenario.broker.orders] == [
        ("ACME", "signal"),
        ("ACME", exit_reason),
    ]
    assert scenario.broker.inventory[foreign.symbol] == foreign
    records = scenario.records()
    assert [r.reason for r in records if r.symbol == foreign.symbol] == [
        "foreign_position_ignored",
        "foreign_position_in_symbol",
    ]


async def test_thousand_dollar_equity_skips_size_below_one_share(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path / "events.jsonl")
    scenario.broker.account_value = 1_000
    async with scenario.running():
        await scenario.news.deliver(scenario.article())
    assert not scenario.broker.orders
    assert [(r.event, r.reason) for r in scenario.records() if r.article_id == 1] == [
        ("skip", "size_below_one_share"),
    ]
