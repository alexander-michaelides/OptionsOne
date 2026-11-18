from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from optionsone.core.clock import SimClock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Bar, Classification, EffectType, EventType, TimeHorizon
from optionsone.features.pipeline import FEATURE_NAMES, build_features

START = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
NEWS = Classification(
    article_id=1,
    ticker="ABC",
    model_id="test",
    prompt_version="1",
    event_type=EventType.EARNINGS,
    effect_type=EffectType.REVENUE,
    time_horizon=TimeHorizon.IMMEDIATE,
    p_positive=0.7,
    p_neutral=0.2,
    p_negative=0.1,
    materiality=0.8,
    novelty=0.8,
    certainty=0.8,
    company_relevance=0.9,
    market_relevance=0.1,
    already_known=0.2,
    knowledge_time=START + timedelta(minutes=30, seconds=14),
)


def history(n: int) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            start=START + timedelta(minutes=i),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100 + i,
        )
        for i in range(n)
        for symbol, close in (("ABC", 100.0 + i), ("SPY", 200.0 + i / 2))
    ]


@given(seconds=st.integers(min_value=31 * 60, max_value=95 * 60))
def test_features_match_physically_truncated_history(seconds: int) -> None:
    clock = SimClock(START + timedelta(seconds=seconds))
    data = history(100)
    full = build_features(NEWS, PointInTimeBars(data, clock))
    truncated = build_features(
        NEWS, PointInTimeBars((b for b in data if b.end <= clock.now()), clock)
    )
    assert full is not None
    assert full == truncated


def test_decision_minute_is_excluded_and_market_windows_align() -> None:
    now = START + timedelta(minutes=40, seconds=37)
    fv = build_features(NEWS, PointInTimeBars(history(50), SimClock(now)))
    assert fv is not None
    assert fv.as_of == now
    assert tuple(fv.values) == FEATURE_NAMES
    assert fv.values["return_1m"] == pytest.approx(139 / 138 - 1)
    assert fv.values["return_5m"] == pytest.approx(139 / 134 - 1)
    assert fv.values["relative_volume"] == pytest.approx(139 / 128.5)
    assert fv.values["return_since_news"] == pytest.approx(139 / 129 - 1)
    assert fv.values["spy_return_5m"] == pytest.approx(219.5 / 217 - 1)
    assert fv.values["market_adjusted_return"] == pytest.approx(139 / 129 - 219.5 / 214.5)


def test_features_require_classification_and_stock_history() -> None:
    data = history(45)
    before_news = SimClock(NEWS.knowledge_time - timedelta(seconds=1))
    assert build_features(NEWS, PointInTimeBars(data, before_news)) is None
    after_news = SimClock(NEWS.knowledge_time + timedelta(minutes=10))
    assert build_features(NEWS, PointInTimeBars(history(30), after_news)) is None
