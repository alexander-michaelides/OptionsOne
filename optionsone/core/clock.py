"""The only module allowed to read wall-clock time."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class LiveClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock:
    """Monotonic simulated clock driven by the replay engine."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock requires a timezone-aware start")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, when: datetime) -> None:
        if when < self._now:
            raise ValueError(f"cannot move clock backwards: {when} < {self._now}")
        self._now = when.astimezone(UTC)
