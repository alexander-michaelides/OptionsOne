from datetime import UTC, datetime, timedelta

from optionsone.core.clock import SimClock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Bar, Position, Side, Signal
from optionsone.execution.costs import CostModel
from optionsone.execution.order_manager import OrderManager, client_order_id
from optionsone.execution.sim_broker import SimBroker


async def test_sim_broker_releases_one_idempotent_fill_only_after_bar_completion() -> None:
    now = datetime(2026, 1, 5, 15, tzinfo=UTC)
    minute = timedelta(minutes=1)
    bar = Bar(symbol="A", start=now, open=100, high=105, low=99, close=104, volume=1000)
    signal = Signal(
        article_id=10,
        ticker="A",
        as_of=now,
        side=Side.SHORT,
        strength=1,
        horizon=minute,
        model_id="test",
    )
    clock = SimClock(now)
    broker = SimBroker(
        PointInTimeBars([bar], clock),
        clock,
        costs=CostModel(half_spread_bps=0, commission_per_share=0),
    )
    manager = OrderManager(broker)
    entry = manager.entry_order(signal, 4, now)

    assert await manager.submit(entry, deadline=now + minute)
    assert not await manager.submit(entry, deadline=now + minute)
    assert not await manager.collect_fills()
    clock.advance_to(bar.end)
    await broker.submit(entry)
    (fill,) = await manager.collect_fills()

    assert (fill.filled_at, fill.price) == (bar.start, 100)
    assert await broker.positions() == (Position(symbol="A", qty=-4, avg_price=100),)
    assert manager.due_symbols(bar.end) == ("A",)
    assert not await manager.collect_fills()


def test_order_id_uses_the_signal_identity_across_rescorings() -> None:
    now = datetime(2026, 1, 5, 15, tzinfo=UTC)
    signal = Signal(
        article_id=10,
        ticker="A",
        as_of=now,
        side=Side.SHORT,
        strength=1,
        horizon=timedelta(minutes=2),
        model_id="test",
    )
    rescored = signal.model_copy(update={"strength": 0.5, "model_id": "new"})
    later = signal.model_copy(update={"as_of": now + timedelta(seconds=1)})

    assert client_order_id(rescored) == client_order_id(signal) != client_order_id(later)
