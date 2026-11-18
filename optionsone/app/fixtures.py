"""Seeded synthetic market for offline end-to-end runs (no network, no credentials).

News articles carry a planted post-publication drift in the headline's direction so the
pipeline has a known signal to recover. Drift starts after the article's knowledge time.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from optionsone.core.types import Article, Bar

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")
BENCHMARK = "SPY"
DELIVERY_LATENCY = timedelta(seconds=30)
_HEADLINES = {
    1: "{s} beats estimates and raises guidance",
    -1: "{s} misses estimates and cuts outlook",
    0: "{s} to present at industry conference",
}


@dataclass(frozen=True, slots=True)
class Fixture:
    bars: tuple[Bar, ...]
    articles: tuple[Article, ...]


def _trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def synthetic_fixture(
    days: int = 10,
    articles_per_day: int = 16,
    drift_per_minute: float = 0.0004,
    drift_minutes: int = 20,
    seed: int = 7,
    start: date = date(2026, 9, 14),
) -> Fixture:
    rng = np.random.default_rng(seed)
    bars: list[Bar] = []
    articles: list[Article] = []
    article_id = 1
    for day in _trading_days(start, days):
        open_utc = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)  # 09:30 EDT
        minutes = 390
        market = rng.normal(0.0, 0.0006, minutes)
        drift = {s: np.zeros(minutes) for s in SYMBOLS}
        for minute in sorted(rng.choice(np.arange(45, 330), articles_per_day, replace=False)):
            symbol = SYMBOLS[int(rng.integers(len(SYMBOLS)))]
            polarity = int(rng.choice([-1, 0, 1]))
            published = open_utc + timedelta(minutes=int(minute), seconds=10)
            articles.append(
                Article(
                    id=article_id,
                    headline=_HEADLINES[polarity].format(s=symbol),
                    symbols=(symbol,),
                    source="synthetic",
                    published_at=published,
                    updated_at=published,
                    knowledge_time=published + DELIVERY_LATENCY,
                )
            )
            article_id += 1
            first = int(minute) + 1  # first bar opening after knowledge time
            drift[symbol][first : first + drift_minutes] += polarity * drift_per_minute
        series = {BENCHMARK: market} | {
            s: 1.2 * market + rng.normal(0.0, 0.001, minutes) + drift[s] for s in SYMBOLS
        }
        for symbol, returns in series.items():
            price = 100.0 + 10 * (sum(map(ord, symbol)) % 7)
            for i, r in enumerate(returns):
                close = price * float(np.exp(r))
                wiggle = abs(float(rng.normal(0.0, 0.0004)))
                bars.append(
                    Bar(
                        symbol=symbol,
                        start=open_utc + timedelta(minutes=i),
                        open=price,
                        high=max(price, close) * (1 + wiggle),
                        low=min(price, close) * (1 - wiggle),
                        close=close,
                        volume=int(rng.integers(2_000, 20_000)),
                    )
                )
                price = close
    return Fixture(bars=tuple(bars), articles=tuple(articles))
