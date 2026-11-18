"""Human-readable progress for long research and backtest runs."""

import sys
from collections.abc import Callable
from time import monotonic
from typing import Protocol


class Progress(Protocol):
    def stage(self, name: str, detail: str = "") -> None:
        """Start a named stage; any open stage is finished first."""
        ...

    def tick(self, done: int, total: int) -> None:
        """Report work done in the current stage (may be throttled)."""
        ...

    def note(self, message: str) -> None:
        """Report a fact about the current stage."""
        ...


class SilentProgress:
    def stage(self, name: str, detail: str = "") -> None:
        pass

    def tick(self, done: int, total: int) -> None:
        pass

    def note(self, message: str) -> None:
        pass


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class ConsoleProgress:
    """Prints `[stage] done/total (pct) rate ETA` at most every `interval` seconds, to stderr."""

    def __init__(
        self,
        write: Callable[[str], None] | None = None,
        *,
        interval: float = 2.0,
        timer: Callable[[], float] = monotonic,
    ) -> None:
        self._write = write if write is not None else self._stderr
        self._interval = interval
        self._timer = timer
        self._name: str | None = None
        self._started = 0.0
        self._last_print = 0.0
        self._run_started = timer()

    @staticmethod
    def _stderr(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    def _finish(self) -> None:
        if self._name is not None:
            self._write(f"[{self._name}] done in {_duration(self._timer() - self._started)}")
            self._name = None

    def stage(self, name: str, detail: str = "") -> None:
        self._finish()
        self._name = name
        self._started = self._last_print = self._timer()
        self._write(f"[{name}] {detail}".rstrip())

    def tick(self, done: int, total: int) -> None:
        now = self._timer()
        if done < total and now - self._last_print < self._interval:
            return
        self._last_print = now
        elapsed = now - self._started
        line = f"[{self._name}] {done:,}/{total:,} ({done / total:.0%})" if total else ""
        if done and elapsed > 0 and done < total:
            rate = done / elapsed
            line += f"  {rate:,.1f}/s  ETA {_duration((total - done) / rate)}"
        if line:
            self._write(line)

    def note(self, message: str) -> None:
        self._write(f"[{self._name or 'run'}] {message}")

    def close(self) -> None:
        self._finish()
        self._write(f"[total] {_duration(self._timer() - self._run_started)}")
