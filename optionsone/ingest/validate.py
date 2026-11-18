from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Protocol

from optionsone.core.types import BAR_DURATION
from optionsone.ingest.calendar import is_regular_session


class BarRow(Protocol):
    """Structural input so malformed prices can be checked before Bar construction."""

    @property
    def symbol(self) -> str: ...
    @property
    def start(self) -> datetime: ...
    @property
    def open(self) -> float: ...
    @property
    def high(self) -> float: ...
    @property
    def low(self) -> float: ...
    @property
    def close(self) -> float: ...
    @property
    def vwap(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class BarTimestamp:
    symbol: str
    start: datetime


@dataclass(frozen=True, slots=True)
class BarInterval:
    """A half-open interval covering missing or unchanged one-minute bars."""

    symbol: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class BarQualityReport:
    gaps: tuple[BarInterval, ...]
    nonpositive_prices: tuple[BarTimestamp, ...]
    duplicates: tuple[BarTimestamp, ...]
    stale_runs: tuple[BarInterval, ...]


def validate_bars(bars: Iterable[BarRow], *, stale_minutes: int = 5) -> BarQualityReport:
    """Report gaps between observations in the same regular session and flat close runs."""
    if stale_minutes < 2:
        raise ValueError("stale_minutes must be at least two")
    gaps: list[BarInterval] = []
    nonpositive: list[BarTimestamp] = []
    duplicates: list[BarTimestamp] = []
    stale: list[BarInterval] = []
    stale_duration = stale_minutes * BAR_DURATION

    for symbol, rows in groupby(
        sorted(bars, key=lambda row: (row.symbol, row.start)), key=lambda row: row.symbol
    ):
        previous: BarRow | None = None
        previous_regular = False
        run_start: datetime | None = None
        for start, same_minute in groupby(rows, key=lambda row: row.start):
            row, *repeats = same_minute
            point = BarTimestamp(symbol, start)
            if repeats:
                duplicates.append(point)
            if any(
                min(item.open, item.high, item.low, item.close) <= 0
                or (item.vwap is not None and item.vwap <= 0)
                for item in (row, *repeats)
            ):
                nonpositive.append(point)
            regular = is_regular_session(start)
            if previous is not None:
                expected = previous.start + BAR_DURATION
                # NYSE regular sessions lie wholly within one UTC date.
                if (
                    previous_regular
                    and regular
                    and previous.start.date() == start.date()
                    and start > expected
                ):
                    gaps.append(BarInterval(symbol, expected, start))
                if start != expected or row.close != previous.close:
                    if run_start is not None and expected - run_start >= stale_duration:
                        stale.append(BarInterval(symbol, run_start, expected))
                    run_start = start
            else:
                run_start = start
            previous, previous_regular = row, regular

        if previous is not None and run_start is not None:
            end = previous.start + BAR_DURATION
            if end - run_start >= stale_duration:
                stale.append(BarInterval(symbol, run_start, end))

    return BarQualityReport(tuple(gaps), tuple(nonpositive), tuple(duplicates), tuple(stale))
