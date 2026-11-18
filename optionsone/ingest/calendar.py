"""NYSE regular-hours calendar for 2025 through 2027."""

from datetime import UTC, date, datetime, time
from typing import Final
from zoneinfo import ZoneInfo

_NEW_YORK: Final = ZoneInfo("America/New_York")
# NYSE Group's published 2025-2027 calendar plus the Carter mourning closure.
_HOLIDAYS: Final = frozenset(
    map(
        date.fromisoformat,
        (
            "2025-01-01",
            "2025-01-09",
            "2025-01-20",
            "2025-02-17",
            "2025-04-18",
            "2025-05-26",
            "2025-06-19",
            "2025-07-04",
            "2025-09-01",
            "2025-11-27",
            "2025-12-25",
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-04-03",
            "2026-05-25",
            "2026-06-19",
            "2026-07-03",
            "2026-09-07",
            "2026-11-26",
            "2026-12-25",
            "2027-01-01",
            "2027-01-18",
            "2027-02-15",
            "2027-03-26",
            "2027-05-31",
            "2027-06-18",
            "2027-07-05",
            "2027-09-06",
            "2027-11-25",
            "2027-12-24",
        ),
    )
)
_HALF_DAYS: Final = frozenset(
    map(
        date.fromisoformat,
        ("2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24", "2027-11-26"),
    )
)


def _is_trading_day(day: date) -> bool:
    if not 2025 <= day.year <= 2027:
        raise ValueError(f"NYSE calendar does not cover {day.year}")
    return day.weekday() < 5 and day not in _HOLIDAYS


def session_open(day: date) -> datetime | None:
    """Return the regular open in UTC, or None on a closed date."""
    if not _is_trading_day(day):
        return None
    return datetime.combine(day, time(9, 30), _NEW_YORK).astimezone(UTC)


def session_close(day: date) -> datetime | None:
    """Return the regular close in UTC, or None on a closed date."""
    if not _is_trading_day(day):
        return None
    return datetime.combine(day, time(13 if day in _HALF_DAYS else 16), _NEW_YORK).astimezone(UTC)


def is_regular_session(ts: datetime) -> bool:
    """Test whether an aware instant is in the half-open regular trading session."""
    if ts.utcoffset() is None:
        raise ValueError("session timestamps must be timezone-aware")
    local = ts.astimezone(_NEW_YORK)
    day = local.date()
    close = time(13 if day in _HALF_DAYS else 16)
    return _is_trading_day(day) and time(9, 30) <= local.time() < close
