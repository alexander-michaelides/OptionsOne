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

    async def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[Bar]:
        if not symbols:
            return []
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
        async with (
            nullcontext(self._client) if self._client is not None else httpx.AsyncClient(timeout=30)
        ) as client:
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
