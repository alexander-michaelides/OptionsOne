"""Strategy interface for point-in-time replay."""

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Protocol

from optionsone.backtest.portfolio import PortfolioSnapshot
from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Signal


class Strategy[Payload](Protocol):
    def events(self) -> Iterator[tuple[datetime, Payload]]:
        """Yield knowledge-time events in chronological order, with stable tie order."""
        ...

    def on_event(
        self, now: datetime, pit_bars: PointInTimeBars, portfolio: PortfolioSnapshot[Payload]
    ) -> Sequence[Signal]:
        """Run once per bar-close or strategy timestamp, with same-time payloads in events."""
        ...
