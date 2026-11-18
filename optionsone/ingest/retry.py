from collections.abc import Awaitable, Callable
from datetime import UTC
from email.utils import parsedate_to_datetime
from threading import Lock
from time import monotonic
from typing import Final

import httpx
from anyio import sleep

from optionsone.core.clock import Clock, LiveClock

type Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """Refill continuously and charge one token for every HTTP attempt."""

    def __init__(
        self,
        requests_per_minute: int = 200,
        *,
        timer: Callable[[], float] = monotonic,
        wait: Sleep = sleep,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self._capacity = float(requests_per_minute)
        self._rate = requests_per_minute / 60
        self._tokens = self._capacity
        self._timer = timer
        self._updated = timer()
        self._wait = wait
        self._lock = Lock()

    async def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._timer()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                delay = (1 - self._tokens) / self._rate
            await self._wait(delay)


_DEFAULT_LIMITER: Final = TokenBucket()


def backoff(attempt: int) -> float:
    return float(2 ** min(attempt, 5))


class RetryPolicy:
    """Share Alpaca's HTTP request budget across historical adapters by default."""

    def __init__(
        self,
        attempts: int = 5,
        *,
        limiter: TokenBucket = _DEFAULT_LIMITER,
        clock: Clock | None = None,
        wait: Sleep = sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self._attempts = attempts
        self._limiter = limiter
        self._clock = clock if clock is not None else LiveClock()
        self._wait = wait

    async def get(
        self, client: httpx.AsyncClient, url: str, params: dict[str, str], headers: dict[str, str]
    ) -> httpx.Response:
        for attempt in range(self._attempts):
            await self._limiter.acquire()
            delay = backoff(attempt)
            last = attempt + 1 == self._attempts
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TransportError:
                if last:
                    raise
            else:
                status = response.status_code
                if last or (status != 429 and not 500 <= status <= 599):
                    return response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    delay = self._retry_after(retry_after, delay)
            await self._wait(delay)
        raise AssertionError("retry loop must return or raise on the final attempt")

    def _retry_after(self, value: str, fallback: float) -> float:
        if value.isascii() and value.isdecimal():
            return float(value)
        try:
            when = parsedate_to_datetime(value)
        except ValueError:
            return fallback
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, (when - self._clock.now()).total_seconds())
