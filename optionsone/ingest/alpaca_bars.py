import re
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Final

import httpx
from pydantic import AwareDatetime, BaseModel, Field

from optionsone.core.types import Bar
from optionsone.ingest.retry import RetryPolicy
from optionsone.ingest.settings import AlpacaSettings

_BARS_URL: Final = "https://data.alpaca.markets/v2/stocks/bars"
_CHUNK: Final = 100  # symbols per request; keeps URLs bounded
# US listings: 1-5 letters with an optional class suffix (BRK.B). News also tags foreign
# tickers (000333, 005930, 1810) that the US bars endpoint rejects with 400.
_US_EQUITY: Final = re.compile(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?")


def is_us_equity_symbol(symbol: str) -> bool:
    return _US_EQUITY.fullmatch(symbol) is not None


class _BarItem(BaseModel):
    timestamp: AwareDatetime = Field(alias="t")
    open: float = Field(alias="o")
    high: float = Field(alias="h")
    low: float = Field(alias="l")
    close: float = Field(alias="c")
    volume: int = Field(alias="v")
    vwap: float | None = Field(default=None, alias="vw")


class _BarsPage(BaseModel):
    bars: dict[str, list[_BarItem]] | None
    next_page_token: str | None = None


class AlpacaHistoricalBars:
    """Retrieve one-minute IEX bars for multiple symbols in one paginated query."""

    def __init__(
        self,
        settings: AlpacaSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._headers = (settings if settings is not None else AlpacaSettings.from_env()).headers()
        self._client = client
        self._retry = retry if retry is not None else RetryPolicy()
        self._unsupported: set[str] = set()

    @property
    def unsupported_symbols(self) -> frozenset[str]:
        """Symbols skipped so far: not US-style, or rejected by Alpaca with 400."""
        return frozenset(self._unsupported)

    async def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[Bar]:
        wanted = list(dict.fromkeys(symbols))
        self._unsupported.update(s for s in wanted if not is_us_equity_symbol(s))
        valid = [s for s in wanted if is_us_equity_symbol(s) and s not in self._unsupported]
        bars: list[Bar] = []
        async with (
            nullcontext(self._client) if self._client is not None else httpx.AsyncClient(timeout=30)
        ) as client:
            for i in range(0, len(valid), _CHUNK):
                bars.extend(await self._fetch_isolating(client, valid[i : i + _CHUNK], start, end))
        return bars

    async def _fetch_isolating(
        self, client: httpx.AsyncClient, symbols: list[str], start: datetime, end: datetime
    ) -> list[Bar]:
        """On 400, split the chunk to find and skip the symbols Alpaca rejects."""
        try:
            return await self._fetch_chunk(client, symbols, start, end)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            if len(symbols) == 1:
                self._unsupported.add(symbols[0])
                return []
            half = len(symbols) // 2
            return [
                *await self._fetch_isolating(client, symbols[:half], start, end),
                *await self._fetch_isolating(client, symbols[half:], start, end),
            ]

    async def _fetch_chunk(
        self, client: httpx.AsyncClient, symbols: list[str], start: datetime, end: datetime
    ) -> list[Bar]:
        params = {
            "symbols": ",".join(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": "1Min",
            "feed": "iex",
            "sort": "asc",
            "limit": "10000",
        }
        bars: list[Bar] = []
        while True:
            response = await self._retry.get(client, _BARS_URL, params, self._headers)
            page = _BarsPage.model_validate_json(response.content)
            for symbol, items in (page.bars or {}).items():
                bars.extend(
                    Bar(
                        symbol=symbol,
                        start=item.timestamp.astimezone(UTC),
                        open=item.open,
                        high=item.high,
                        low=item.low,
                        close=item.close,
                        volume=item.volume,
                        vwap=item.vwap,
                    )
                    for item in items
                )
            if not page.next_page_token:
                return bars
            params["page_token"] = page.next_page_token
