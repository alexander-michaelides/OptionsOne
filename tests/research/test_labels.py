from datetime import UTC, datetime, timedelta

import pytest

from optionsone.core.types import Bar
from optionsone.research.labels import BarrierSpec, forward_returns, triple_barrier_label

T0 = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def bar(minute: int, symbol: str, open_: float, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        start=T0 + minute * MINUTE,
        open=open_,
        close=close,
        high=max(open_, close),
        low=min(open_, close),
        volume=100,
    )


@pytest.mark.parametrize(
    ("latency", "entry_minute", "open_price"), [(timedelta(0), 1, 100), (2 * MINUTE, 2, 104)]
)
def test_forward_entry_open_is_strictly_after_decision(
    latency: timedelta, entry_minute: int, open_price: int
) -> None:
    stock = [bar(0, "ABC", 50, 60), bar(1, "ABC", 100, 102), bar(2, "ABC", 104, 106)]
    labels = forward_returns(T0, "ABC", stock, entry_latency=latency)
    assert set(labels) == {1}
    assert labels[1].entry_time == T0 + entry_minute * MINUTE
    assert labels[1].end_time == T0 + (entry_minute + 1) * MINUTE
    assert labels[1].raw_return == pytest.approx(stock[entry_minute].close / open_price - 1)


def test_adjustment_uses_aligned_entry_and_exact_exit() -> None:
    data = [
        bar(1, "ABC", 100, 101),
        bar(5, "ABC", 105, 112),
        bar(6, "ABC", 113, 120),
        bar(1, "SPY", 200, 202),
        bar(5, "SPY", 205, 208),
    ]
    labels = forward_returns(T0, "ABC", data, horizons=(4, 5, 6))
    assert set(labels) == {5, 6}
    assert labels[5].market_adjusted_return == pytest.approx(0.08)
    assert labels[6].market_adjusted_return is None


def test_triple_barrier_scales_to_volatility_and_stops_first_on_tie() -> None:
    entry = bar(1, "ABC", 100, 100).model_copy(update={"high": 103, "low": 98})
    narrow = triple_barrier_label(T0, "ABC", [entry], BarrierSpec(0.01, max_holding=1))
    wide = triple_barrier_label(T0, "ABC", [entry], BarrierSpec(0.03, max_holding=1))
    assert narrow is not None
    assert wide is not None
    assert narrow.label == -1
    assert narrow.end_time == T0 + 2 * MINUTE
    assert narrow.raw_return == pytest.approx(-0.015)
    assert wide.label == 0
    assert triple_barrier_label(T0, "ABC", [entry], BarrierSpec(0.03)) is None
