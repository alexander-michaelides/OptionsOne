from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pytest

from optionsone.ingest.calendar import is_regular_session, session_close, session_open
from optionsone.ingest.validate import BarInterval, BarTimestamp, validate_bars


@pytest.mark.parametrize(
    "day",
    [date(2025, 1, 9), date(2025, 7, 4), date(2026, 7, 3), date(2027, 12, 24)],
)
def test_nyse_closed_on_observed_holidays(day: date) -> None:
    assert session_open(day) is None
    assert session_close(day) is None
    assert not is_regular_session(datetime(day.year, day.month, day.day, 16, tzinfo=UTC))


@pytest.mark.parametrize(
    ("day", "open_hour", "close_hour"),
    [(date(2025, 7, 3), 13, 17), (date(2027, 11, 26), 14, 18)],
)
def test_half_day_excludes_close_and_uses_new_york_time(
    day: date, open_hour: int, close_hour: int
) -> None:
    opening = datetime(day.year, day.month, day.day, open_hour, 30, tzinfo=UTC)
    closing = datetime(day.year, day.month, day.day, close_hour, tzinfo=UTC)
    assert session_open(day) == opening
    assert session_close(day) == closing
    assert not is_regular_session(opening - timedelta(minutes=1))
    assert is_regular_session(opening)
    assert is_regular_session(closing - timedelta(minutes=1))
    assert not is_regular_session(closing)


@dataclass(frozen=True, slots=True)
class RawBar:
    start: datetime
    symbol: str = "AAPL"
    open: float = 200
    high: float = 200
    low: float = 200
    close: float = 200
    vwap: float | None = None


def test_bar_quality_finds_intraday_gap_and_separates_flat_runs() -> None:
    opening = datetime(2025, 7, 3, 13, 30, tzinfo=UTC)
    one = timedelta(minutes=1)
    post_close = datetime(2025, 7, 3, 17, 1, tzinfo=UTC)
    next_session = datetime(2025, 7, 7, 13, 30, tzinfo=UTC)
    rows = [
        RawBar(next_session),
        RawBar(post_close),
        RawBar(opening + 4 * one),
        RawBar(opening + 3 * one),
        RawBar(opening),
        RawBar(opening + one),
        RawBar(opening + one, low=0),
        RawBar(opening + 2 * one, "MSFT"),
        RawBar(opening + 3 * one, "MSFT", vwap=-1),
    ]

    report = validate_bars(rows, stale_minutes=2)

    assert report.gaps == (BarInterval("AAPL", opening + 2 * one, opening + 3 * one),)
    assert report.duplicates == (BarTimestamp("AAPL", opening + one),)
    assert report.nonpositive_prices == (
        BarTimestamp("AAPL", opening + one),
        BarTimestamp("MSFT", opening + 3 * one),
    )
    assert report.stale_runs == (
        BarInterval("AAPL", opening, opening + 2 * one),
        BarInterval("AAPL", opening + 3 * one, opening + 5 * one),
        BarInterval("MSFT", opening + 2 * one, opening + 4 * one),
    )
