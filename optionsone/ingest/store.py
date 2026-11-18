from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Final
from urllib.parse import quote

import polars as pl

from optionsone.core.types import Article, Bar

_UTC_DATETIME: Final = pl.Datetime("us", "UTC")
_ARTICLE_SCHEMA: Final = pl.Schema(
    {
        "id": pl.Int64,
        "headline": pl.String,
        "summary": pl.String,
        "content": pl.String,
        "source": pl.String,
        "symbols": pl.List(pl.String),
        "published_at": _UTC_DATETIME,
        "updated_at": _UTC_DATETIME,
        "knowledge_time": _UTC_DATETIME,
    }
)
_BAR_SCHEMA: Final = pl.Schema(
    {
        "symbol": pl.String,
        "start": _UTC_DATETIME,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "vwap": pl.Float64,
    }
)


def _append(partition: Path, incoming: pl.DataFrame, keys: Sequence[str]) -> int:
    incoming = incoming.unique(subset=keys, keep="first", maintain_order=True)
    paths = sorted(partition.glob("*.parquet"))
    if paths:
        existing = pl.read_parquet(paths, hive_partitioning=False).select(keys)
        incoming = incoming.join(existing, on=keys, how="anti")
    if incoming.is_empty():
        return 0

    partition.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=partition) as directory:
        temp = Path(directory)
        incoming.write_parquet(temp / "rows.parquet")
        (temp / "rows.parquet").replace(partition / f"{temp.name}.parquet")
    return incoming.height


class ParquetStore:
    """Hive-partitioned Parquet store. Use one writer instance per root."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._write_lock = Lock()

    def write_articles(self, articles: Iterable[Article]) -> int:
        """Append unseen revisions, preserving the first stored receipt for each key."""
        partitions: defaultdict[Path, list[Article]] = defaultdict(list)
        for article in articles:
            # Partition on a field in the deduplication key, even for late arrivals.
            partition = self._root / "news" / f"date={article.updated_at.date()}"
            partitions[partition].append(article)
        with self._write_lock:
            return sum(
                _append(
                    partition,
                    pl.from_dicts((row.model_dump() for row in rows), schema=_ARTICLE_SCHEMA),
                    ["id", "updated_at"],
                )
                for partition, rows in partitions.items()
            )

    def write_bars(self, bars: Iterable[Bar]) -> int:
        """Append unseen symbol/minute pairs and return the number added."""
        partitions: defaultdict[Path, list[Bar]] = defaultdict(list)
        for bar in bars:
            partitions[self._bar_symbol(bar.symbol) / f"date={bar.start.date()}"].append(bar)
        with self._write_lock:
            return sum(
                _append(
                    partition,
                    pl.from_dicts((row.model_dump() for row in rows), schema=_BAR_SCHEMA),
                    ["symbol", "start"],
                )
                for partition, rows in partitions.items()
            )

    def load_articles(self, start: datetime, end: datetime) -> list[Article]:
        """Load all revisions with knowledge time in [start, end), oldest first."""
        paths = sorted((self._root / "news").glob("date=*/*.parquet"))
        if not paths:
            return []
        rows = (
            pl.scan_parquet(paths, hive_partitioning=False)
            .filter(pl.col("knowledge_time").is_between(start, end, closed="left"))
            .sort("knowledge_time", "id", "updated_at")
            .collect()
        )
        return [Article.model_validate(row) for row in rows.iter_rows(named=True)]

    def load_bars(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[Bar]:
        """Load symbol/minute pairs with bar start in [start, end)."""
        if not symbols or start >= end:
            return []
        first_day, last_day = start.astimezone(UTC).date(), end.astimezone(UTC).date()
        days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
        paths = [
            path
            for symbol in sorted(set(symbols))
            for day in days
            for path in (self._bar_symbol(symbol) / f"date={day}").glob("*.parquet")
        ]
        if not paths:
            return []
        rows = (
            pl.scan_parquet(paths, hive_partitioning=False)
            .filter(pl.col("start").is_between(start, end, closed="left"))
            .sort("symbol", "start")
            .collect()
        )
        return [Bar.model_validate(row) for row in rows.iter_rows(named=True)]

    def _bar_symbol(self, symbol: str) -> Path:
        return self._root / "bars" / f"symbol={quote(symbol, safe='')}"
