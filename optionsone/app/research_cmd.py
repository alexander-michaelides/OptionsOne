"""Application workflows for historical ingestion and stored-data research."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from json import dumps
from pathlib import Path
from typing import Final

from anyio import Path as AsyncPath
from anyio.to_thread import run_sync

from optionsone.app.offline import (
    build_dataset,
    classify_articles,
    evaluate_and_fit,
    purge_dataset,
)
from optionsone.core.ports import Classifier, HistoricalBars, HistoricalNews
from optionsone.core.types import Bar
from optionsone.ingest.calendar import session_close, session_open
from optionsone.ingest.store import ParquetStore
from optionsone.ingest.validate import BarQualityReport, validate_bars
from optionsone.research.evaluate import EvaluationReport, passes_promotion_gate
from optionsone.signals.model import LinearSignalModel

_BENCHMARK: Final = "SPY"


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """Selected input row counts, inserted row counts, and fetched-bar quality."""

    articles_fetched: int
    bars_fetched: int
    articles_added: int
    bars_added: int
    validation: BarQualityReport


@dataclass(frozen=True, slots=True)
class ResearchReport:
    evaluation: EvaluationReport
    model: LinearSignalModel
    passes_promotion_gate: bool

    @property
    def model_id(self) -> str:
        return self.model.model_id


async def backfill(
    store_root: Path,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    news: HistoricalNews,
    bars: HistoricalBars,
) -> BackfillReport:
    """Store publication-window news and regular-session bars in [start, end)."""
    universe = set(symbols)
    articles = [
        article
        for article in await news.fetch(start, end)
        if start <= article.published_at < end
        and (not universe or not universe.isdisjoint(article.symbols))
    ]
    mentioned = {symbol for article in articles for symbol in article.symbols}
    requested = sorted((mentioned & universe if universe else mentioned) | {_BENCHMARK})
    fetched: list[Bar] = []
    day = start.astimezone(UTC).date()
    # Each NYSE regular session opens and closes on the same UTC date.
    while datetime.combine(day, time.min, UTC) < end:
        opening, closing = session_open(day), session_close(day)
        day += timedelta(days=1)
        if opening is None or closing is None:
            continue
        lower, upper = max(start, opening), min(end, closing)
        if lower < upper:
            fetched.extend(
                bar
                for bar in await bars.fetch(requested, lower, upper)
                if lower <= bar.start < upper
            )

    store = ParquetStore(store_root)
    validation = await run_sync(validate_bars, fetched)
    return BackfillReport(
        articles_fetched=len(articles),
        bars_fetched=len(fetched),
        articles_added=await run_sync(store.write_articles, articles),
        bars_added=await run_sync(store.write_bars, fetched),
        validation=validation,
    )


async def research(
    store_root: Path,
    start: datetime,
    end: datetime,
    classifier: Classifier,
    *,
    model_out: Path,
    report_out: Path,
    registry_path: Path,
    n_splits: int = 5,
) -> ResearchReport:
    """Evaluate stored data and export a model fitted using outcomes known before end."""
    store = ParquetStore(store_root)
    articles = await run_sync(store.load_articles, start, end)
    symbols = sorted({symbol for article in articles for symbol in article.symbols} | {_BENCHMARK})
    bars = await run_sync(store.load_bars, symbols, start, end)
    classifications = await classify_articles(articles, classifier)
    dataset = purge_dataset(await run_sync(build_dataset, classifications, bars), end)

    await AsyncPath(registry_path.parent).mkdir(parents=True, exist_ok=True)
    evaluation, model = await run_sync(evaluate_and_fit, dataset, registry_path, n_splits)
    passed = passes_promotion_gate(evaluation)
    report_json = dumps(
        {**asdict(evaluation), "passes_promotion_gate": passed}, indent=2, allow_nan=False
    )
    for path, payload in ((model_out, model.to_json()), (report_out, report_json)):
        await AsyncPath(path.parent).mkdir(parents=True, exist_ok=True)
        await AsyncPath(path).write_text(payload + "\n", encoding="utf-8")
    return ResearchReport(evaluation, model, passed)
