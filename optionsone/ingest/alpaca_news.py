from collections.abc import AsyncGenerator
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from json import dumps
from typing import Final

import httpx
from anyio import fail_after, sleep
from pydantic import AwareDatetime, BaseModel, JsonValue, TypeAdapter
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from optionsone.core.clock import Clock, LiveClock
from optionsone.core.types import Article
from optionsone.ingest.retry import RetryPolicy, Sleep, backoff
from optionsone.ingest.settings import AlpacaSettings

_NEWS_URL: Final = "https://data.alpaca.markets/v1beta1/news"
_STREAM_URL: Final = "wss://stream.data.alpaca.markets/v1beta1/news"
_EVENTS: Final = TypeAdapter(list[dict[str, JsonValue]])


class _NewsItem(BaseModel):
    id: int
    headline: str
    summary: str | None = None
    content: str | None = None
    source: str | None = None
    author: str | None = None
    symbols: tuple[str, ...]
    created_at: AwareDatetime
    updated_at: AwareDatetime

    def article(self, knowledge_time: datetime) -> Article:
        return Article(
            id=self.id,
            headline=self.headline,
            summary=self.summary or "",
            content=self.content or "",
            source=self.source or self.author or "",
            symbols=self.symbols,
            published_at=self.created_at.astimezone(UTC),
            updated_at=self.updated_at.astimezone(UTC),
            knowledge_time=knowledge_time,
        )


class _NewsPage(BaseModel):
    news: list[_NewsItem]
    next_page_token: str | None = None


class AlpacaHistoricalNews:
    """Fetch the revisions Alpaca returns for the requested publication window."""

    def __init__(
        self,
        settings: AlpacaSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        retry: RetryPolicy | None = None,
        delivery_latency: timedelta = timedelta(seconds=30),
    ) -> None:
        if delivery_latency < timedelta(0):
            raise ValueError("delivery_latency must be nonnegative")
        self._headers = (settings if settings is not None else AlpacaSettings.from_env()).headers()
        self._client = client
        self._retry = retry if retry is not None else RetryPolicy()
        self._latency = delivery_latency

    async def fetch(self, start: datetime, end: datetime) -> list[Article]:
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort": "asc",
            "include_content": "true",
            "limit": "50",
        }
        articles: list[Article] = []
        async with (
            nullcontext(self._client) if self._client is not None else httpx.AsyncClient(timeout=30)
        ) as client:
            while True:
                response = await self._retry.get(client, _NEWS_URL, params, self._headers)
                page = _NewsPage.model_validate_json(response.content)
                for item in page.news:
                    revised_at = max(item.created_at, item.updated_at).astimezone(UTC)
                    articles.append(item.article(revised_at + self._latency))
                if not page.next_page_token:
                    return articles
                params["page_token"] = page.next_page_token


class NewsStreamError(RuntimeError):
    """Alpaca rejected the stream authentication or subscription."""


class AlpacaNewsStream:
    """Subscribe to all Alpaca news and stamp each frame when it is received."""

    def __init__(
        self,
        settings: AlpacaSettings | None = None,
        *,
        clock: Clock | None = None,
        wait: Sleep = sleep,
        url: str = _STREAM_URL,
    ) -> None:
        self._settings = settings if settings is not None else AlpacaSettings.from_env()
        self._clock = clock if clock is not None else LiveClock()
        self._wait = wait
        self._url = url

    async def stream(self) -> AsyncGenerator[Article]:
        auth = dumps(
            {
                "action": "auth",
                "key": self._settings.api_key.get_secret_value(),
                "secret": self._settings.secret_key.get_secret_value(),
            }
        )
        attempt = 0
        while True:
            try:
                async with connect(self._url) as socket:
                    with fail_after(10):
                        await socket.send(auth)
                        authenticated = False
                        while not authenticated:
                            for event in _EVENTS.validate_json(await socket.recv()):
                                if event.get("T") == "error":
                                    raise NewsStreamError(f"Alpaca authentication failed: {event}")
                                if (
                                    event.get("T") == "success"
                                    and event.get("msg") == "authenticated"
                                ):
                                    authenticated = True
                        await socket.send(dumps({"action": "subscribe", "news": ["*"]}))
                    async for frame in socket:
                        received_at = self._clock.now()
                        for event in _EVENTS.validate_json(frame):
                            if event.get("T") == "error":
                                raise NewsStreamError(f"Alpaca news stream error: {event}")
                            if event.get("T") == "n":
                                attempt = 0
                                yield _NewsItem.model_validate(event).article(received_at)
            except InvalidStatus as exc:
                status = exc.response.status_code
                if status != 429 and not 500 <= status <= 599:
                    raise
            except ConnectionClosed, EOFError, OSError:
                pass
            await self._wait(backoff(attempt))
            attempt = min(attempt + 1, 5)
