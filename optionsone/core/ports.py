"""Interfaces between layers. Adapters implement these; core logic depends only on them."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol

from optionsone.core.types import Article, Bar, Classification, Fill, OrderIntent, Position


class NewsSource(Protocol):
    def stream(self) -> AsyncIterator[Article]: ...


class HistoricalNews(Protocol):
    async def fetch(self, start: datetime, end: datetime) -> Sequence[Article]: ...


class HistoricalBars(Protocol):
    async def fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[Bar]: ...


class Classifier(Protocol):
    @property
    def model_id(self) -> str: ...

    async def classify(self, article: Article, ticker: str) -> Classification: ...


class Broker(Protocol):
    """Order sink. Implementations: SimBroker (backtest) and AlpacaPaperBroker (paper only)."""

    async def submit(self, order: OrderIntent) -> None: ...

    async def fills(self) -> Sequence[Fill]: ...

    async def positions(self) -> Sequence[Position]: ...

    async def cancel(self, client_order_ids: Sequence[str]) -> None:
        """Cancel only the named orders; never account-wide (the account may hold others)."""
        ...
