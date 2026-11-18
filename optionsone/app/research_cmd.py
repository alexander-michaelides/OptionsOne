"""Application workflows for historical ingestion and stored-data research."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from json import dumps, loads
from pathlib import Path
from typing import Final

from anyio import Path as AsyncPath
from anyio.to_thread import run_sync

from optionsone.app.offline import (
    OfflineRun,
    build_dataset,
    classify_articles,
    evaluate_and_fit,
    purge_dataset,
    run_offline,
)
from optionsone.core.clock import Clock, LiveClock
from optionsone.core.ports import Classifier, HistoricalBars, HistoricalNews
from optionsone.core.types import Article, Bar
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


@dataclass(frozen=True, slots=True)
class DayProgress:
    """One UTC day of a backfill; `resumed` means an earlier run already completed it."""

    day: date
    articles_added: int
    bars_added: int
    resumed: bool


type OnDay = Callable[[DayProgress], None]


class _Manifest:
    """Append-only record of fully elapsed day windows already written for a universe."""

    def __init__(self, path: Path, universe: set[str]) -> None:
        self._path = path
        self._universe = ",".join(sorted(universe)) or "*"
        self._done: set[tuple[str, str]] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = loads(line)
                if entry["universe"] == self._universe:
                    self._done.add((entry["start"], entry["end"]))

    def done(self, lower: datetime, upper: datetime) -> bool:
        return (lower.isoformat(), upper.isoformat()) in self._done

    def mark(self, lower: datetime, upper: datetime) -> None:
        entry = {"universe": self._universe, "start": lower.isoformat(), "end": upper.isoformat()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(entry) + "\n")
        self._done.add((entry["start"], entry["end"]))


async def backfill(
    store_root: Path,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    news: HistoricalNews,
    bars: HistoricalBars,
    on_day: OnDay | None = None,
    clock: Clock | None = None,
) -> BackfillReport:
    """Store publication-window news and regular-session bars in [start, end), day by day.

    Each UTC day is fetched, validated and written before the next one starts, so memory stays
    bounded and an interrupted run resumes: fully elapsed days already written for the same
    universe are skipped on the next run.
    """
    universe = set(symbols)
    store = ParquetStore(store_root)
    manifest = await run_sync(_Manifest, store_root / "_backfill_done.jsonl", universe)
    now = (clock if clock is not None else LiveClock()).now()
    totals = [0, 0, 0, 0]  # articles fetched/added, bars fetched/added
    quality: list[BarQualityReport] = []
    day = start.astimezone(UTC).date()
    while (day_start := datetime.combine(day, time.min, UTC)) < end:
        lower, upper = max(start, day_start), min(end, day_start + timedelta(days=1))
        day += timedelta(days=1)
        if manifest.done(lower, upper):
            if on_day is not None:
                on_day(DayProgress(lower.date(), 0, 0, resumed=True))
            continue
        articles = [
            article
            for article in await news.fetch(lower, upper)
            if lower <= article.published_at < upper
            and (not universe or not universe.isdisjoint(article.symbols))
        ]
        mentioned = {symbol for article in articles for symbol in article.symbols}
        requested = sorted((mentioned & universe if universe else mentioned) | {_BENCHMARK})
        fetched: list[Bar] = []
        # Each NYSE regular session opens and closes on the same UTC date.
        opening, closing = session_open(lower.date()), session_close(lower.date())
        if opening is not None and closing is not None:
            bar_lower, bar_upper = max(lower, opening), min(upper, closing)
            if bar_lower < bar_upper:
                fetched = [
                    bar
                    for bar in await bars.fetch(requested, bar_lower, bar_upper)
                    if bar_lower <= bar.start < bar_upper
                ]
        quality.append(await run_sync(validate_bars, fetched))
        articles_added = await run_sync(store.write_articles, articles)
        bars_added = await run_sync(store.write_bars, fetched)
        for i, value in enumerate((len(articles), articles_added, len(fetched), bars_added)):
            totals[i] += value
        if upper <= now:  # only windows that can no longer change are marked complete
            await run_sync(manifest.mark, lower, upper)
        if on_day is not None:
            on_day(DayProgress(lower.date(), articles_added, bars_added, resumed=False))

    return BackfillReport(
        articles_fetched=totals[0],
        bars_fetched=totals[2],
        articles_added=totals[1],
        bars_added=totals[3],
        validation=BarQualityReport(
            gaps=tuple(g for q in quality for g in q.gaps),
            nonpositive_prices=tuple(p for q in quality for p in q.nonpositive_prices),
            duplicates=tuple(d for q in quality for d in q.duplicates),
            stale_runs=tuple(r for q in quality for r in q.stale_runs),
        ),
    )


class EmptyStoreError(ValueError):
    """No stored news matched the requested window and universe."""


async def load_store(
    store_root: Path, start: datetime, end: datetime, symbols: Sequence[str] = ()
) -> tuple[list[Article], list[Bar]]:
    """Stored news in [start, end), optionally narrowed to a universe, plus bars it needs."""
    store = ParquetStore(store_root)
    articles = await run_sync(store.load_articles, start, end)
    universe = {symbol.upper() for symbol in symbols}
    if universe:
        articles = [
            article.model_copy(
                update={"symbols": tuple(s for s in article.symbols if s in universe)}
            )
            for article in articles
            if not universe.isdisjoint(article.symbols)
        ]
    needed = sorted({symbol for article in articles for symbol in article.symbols} | {_BENCHMARK})
    bars = await run_sync(store.load_bars, needed, start, end)
    return articles, bars


async def backtest_store(
    store_root: Path,
    start: datetime,
    end: datetime,
    classifier: Classifier,
    *,
    registry_path: Path,
    symbols: Sequence[str] = (),
    train_fraction: float = 0.5,
    n_splits: int = 4,
    initial_cash: float = 100_000.0,
) -> OfflineRun:
    """Fit on the earlier part of the stored window, replay the rest through the engine."""
    articles, bars = await load_store(store_root, start, end, symbols)
    if not articles:
        raise EmptyStoreError(
            f"no stored news in [{start:%Y-%m-%d}, {end:%Y-%m-%d}) for this universe; "
            "run backfill first"
        )
    await AsyncPath(registry_path.parent).mkdir(parents=True, exist_ok=True)
    return await run_offline(
        articles,
        bars,
        classifier,
        registry_path,
        train_fraction=train_fraction,
        n_splits=n_splits,
        initial_cash=initial_cash,
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
    articles, bars = await load_store(store_root, start, end)
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
