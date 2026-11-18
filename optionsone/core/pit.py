"""Point-in-time guarded data access.

All research and live components read bars and articles through these views,
which hide anything whose knowledge time is after the clock. This is the single
enforcement point for "no future information".
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from datetime import datetime

from optionsone.core.clock import Clock
from optionsone.core.types import Article, Bar


class LookaheadError(RuntimeError):
    """Raised when code asks for data beyond the current knowledge time."""


class PointInTimeBars:
    """Sorted per-symbol bars, exposed only up to `clock.now()` by bar *end*."""

    def __init__(self, bars: Iterable[Bar], clock: Clock) -> None:
        self._clock = clock
        by_symbol: dict[str, list[Bar]] = {}
        for bar in bars:
            by_symbol.setdefault(bar.symbol, []).append(bar)
        self._bars = {s: sorted(b, key=lambda x: x.start) for s, b in by_symbol.items()}
        self._ends = {s: [b.end for b in bs] for s, bs in self._bars.items()}

    def now(self) -> datetime:
        """The knowledge time this view is filtered at."""
        return self._clock.now()

    def symbols(self) -> frozenset[str]:
        return frozenset(self._bars)

    def bars_asof(self, symbol: str, since: datetime | None = None) -> Sequence[Bar]:
        """Bars with `end <= now` (and `start >= since` when given), oldest first."""
        bars = self._bars.get(symbol, [])
        ends = self._ends.get(symbol, [])
        hi = bisect.bisect_right(ends, self._clock.now())
        lo = 0 if since is None else bisect.bisect_left([b.start for b in bars[:hi]], since)
        return bars[lo:hi]

    def last_bar(self, symbol: str) -> Bar | None:
        visible = self.bars_asof(symbol)
        return visible[-1] if visible else None

    def next_bar_after(self, symbol: str, when: datetime) -> Bar | None:
        """First bar opening at or after `when`, for fills. Must already be complete."""
        for bar in self._bars.get(symbol, []):
            if bar.start >= when:
                if bar.end > self._clock.now():
                    raise LookaheadError(f"bar {bar.start} for {symbol} not complete at now")
                return bar
        return None


class PointInTimeNews:
    """Articles visible only once their knowledge_time has passed; latest revision wins."""

    def __init__(self, articles: Iterable[Article], clock: Clock) -> None:
        self._clock = clock
        self._articles = sorted(articles, key=lambda a: (a.knowledge_time, a.id))

    def visible(self) -> list[Article]:
        now = self._clock.now()
        latest: dict[int, Article] = {}
        for article in self._articles:
            if article.knowledge_time > now:
                break
            latest[article.id] = article
        return sorted(latest.values(), key=lambda a: (a.knowledge_time, a.id))
