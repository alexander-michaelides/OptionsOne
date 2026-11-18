"""Recent completed market data for paper-trading decisions."""

from collections.abc import Iterable
from datetime import datetime, timedelta

from optionsone.core.clock import Clock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.ports import HistoricalBars
from optionsone.core.types import Bar


class LiveBarCache:
    """Cache a rolling bar window per symbol set for the current minute."""

    def __init__(self, bars: HistoricalBars, clock: Clock, *, lookback_minutes: int = 45) -> None:
        if lookback_minutes <= 0:
            raise ValueError("lookback_minutes must be positive")
        self._bars = bars
        self._clock = clock
        self._lookback = timedelta(minutes=lookback_minutes)
        self._minute: datetime | None = None
        self._cached: dict[tuple[str, ...], tuple[Bar, ...]] = {}

    async def refresh(self, symbols: Iterable[str]) -> PointInTimeBars:
        """Fetch the window when needed and expose it through the injected clock."""
        requested = tuple(sorted({*symbols, "SPY"}))
        minute = self._clock.now().replace(second=0, microsecond=0)
        if self._minute != minute:
            self._minute = minute
            self._cached.clear()
        bars = self._cached.get(requested)
        if bars is None:
            # Exclude the current minute's start even for an inclusive REST end.
            end = minute - timedelta(microseconds=1)
            bars = tuple(await self._bars.fetch(requested, minute - self._lookback, end))
            if self._minute == minute:
                self._cached[requested] = bars
        return PointInTimeBars(bars, self._clock)
