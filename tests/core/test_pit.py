from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from optionsone.core.clock import SimClock
from optionsone.core.pit import LookaheadError, PointInTimeBars, PointInTimeNews
from optionsone.core.types import Article, Bar

T0 = datetime(2026, 9, 21, 13, 30, tzinfo=UTC)


def _bars(n: int, symbol: str = "AAPL") -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            start=T0 + timedelta(minutes=i),
            open=100,
            high=200,
            low=99,
            close=100 + i * 0.01,
            volume=1000,
        )
        for i in range(n)
    ]


@given(n=st.integers(1, 120), offset_s=st.integers(-60, 8000))
def test_bars_asof_never_returns_incomplete_bar(n: int, offset_s: int) -> None:
    clock = SimClock(T0 + timedelta(seconds=offset_s)) if offset_s >= 0 else SimClock(T0)
    view = PointInTimeBars(_bars(n), clock)
    visible = view.bars_asof("AAPL")
    assert all(b.end <= clock.now() for b in visible)
    hidden = n - len(visible)
    assert hidden == sum(1 for b in _bars(n) if b.end > clock.now())


def test_bar_containing_news_minute_is_hidden() -> None:
    # News at 13:30:20 -> the 13:30 bar closes at 13:31 and must not be visible.
    clock = SimClock(T0 + timedelta(seconds=20))
    assert PointInTimeBars(_bars(5), clock).bars_asof("AAPL") == []


def test_next_bar_after_refuses_incomplete_bar() -> None:
    clock = SimClock(T0 + timedelta(seconds=30))
    with pytest.raises(LookaheadError):
        PointInTimeBars(_bars(5), clock).next_bar_after("AAPL", T0)


def test_news_hides_future_revisions() -> None:
    later = T0 + timedelta(hours=1)
    v1 = Article(
        id=1, symbols=("AAPL",), published_at=T0, headline="v1", updated_at=T0, knowledge_time=T0
    )
    v2 = Article(
        id=1,
        symbols=("AAPL",),
        published_at=T0,
        headline="v2",
        updated_at=later,
        knowledge_time=later,
    )
    clock = SimClock(T0 + timedelta(minutes=5))
    news = PointInTimeNews([v2, v1], clock)
    assert [a.headline for a in news.visible()] == ["v1"]
    clock.advance_to(T0 + timedelta(hours=2))
    assert [a.headline for a in news.visible()] == ["v2"]


def test_clock_is_monotonic() -> None:
    clock = SimClock(T0)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_to(T0 - timedelta(seconds=1))
