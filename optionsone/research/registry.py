"""Append-only experiment history for one strategy search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from optionsone.core.types import Frozen

type Parameter = str | int | float | bool | tuple[str, ...]


class TrialRecord(Frozen):
    trial_id: str
    kind: Literal["trial_started", "fit_started", "fit_completed", "trial_completed"]
    data_hash: str
    params: dict[str, Parameter]
    fold: int | None = None
    model_id: str | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrialRegistry:
    """Keep all attempted candidates in one JSONL file, including incomplete runs."""

    path: Path

    def append(self, record: TrialRecord) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")

    def records(self) -> tuple[TrialRecord, ...]:
        try:
            stream = self.path.open(encoding="utf-8")
        except FileNotFoundError:
            return ()
        with stream:
            return tuple(TrialRecord.model_validate_json(line) for line in stream)
