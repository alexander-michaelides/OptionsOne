"""Append-only audit events for the paper runner."""

from pathlib import Path
from typing import Literal

from optionsone.core.types import Fill, Frozen, OrderIntent, Signal, UtcDatetime


class JournalRecord(Frozen):
    timestamp: UtcDatetime
    event: Literal["skip", "entry", "exit", "fill", "cancel", "error"]
    reason: str
    article_id: int | None = None
    updated_at: UtcDatetime | None = None
    symbol: str | None = None
    signal: Signal | None = None
    order: OrderIntent | None = None
    fill: Fill | None = None
    error_type: str | None = None


class Journal:
    """Write one UTF-8 JSON object per line to the supplied path."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, record: JournalRecord) -> None:
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json(exclude_none=True) + "\n")
