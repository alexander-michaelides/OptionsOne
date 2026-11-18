"""Offline research + backtest pipeline: classify, build PIT features, evaluate, replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from optionsone.app.progress import Progress, SilentProgress
from optionsone.app.strategy import NewsStrategy
from optionsone.backtest.engine import BacktestResult, run_backtest
from optionsone.classify.batcher import ClassificationFailure, classify_many
from optionsone.classify.cache import CachedClassifier
from optionsone.core.clock import SimClock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.ports import Classifier
from optionsone.core.types import Article, Bar, Classification, FeatureVector
from optionsone.features.pipeline import build_features
from optionsone.ingest.calendar import is_regular_session
from optionsone.research.evaluate import EvaluationReport, EventKey, Outcome, evaluate
from optionsone.research.labels import BarIndex, forward_returns
from optionsone.research.registry import TrialRegistry
from optionsone.signals.model import LinearSignalModel

HORIZON_MINUTES = 15
_WARMUP = timedelta(hours=1)  # enough completed bars for the 31-close feature window
_EXCHANGE_TZ = ZoneInfo("America/New_York")
SILENT = SilentProgress()


@dataclass(frozen=True, slots=True)
class Dataset:
    features: tuple[FeatureVector, ...]
    labels: dict[EventKey, Outcome]


@dataclass(frozen=True, slots=True)
class OfflineRun:
    n_classifications: int
    research_train: EvaluationReport
    model_id: str
    replay_start: datetime
    backtest: BacktestResult


async def classify_articles(
    articles: Sequence[Article],
    classifier: Classifier,
    concurrency: int = 8,
    progress: Progress = SILENT,
) -> list[Classification]:
    items = [(article, ticker) for article in articles for ticker in article.symbols]
    progress.stage(
        "classify", f"{len(items):,} article/ticker pairs with model '{classifier.model_id}'"
    )
    results = await classify_many(classifier, items, concurrency=concurrency, on_done=progress.tick)
    if isinstance(classifier, CachedClassifier):
        progress.note(
            f"{classifier.hits:,} answered from cache, {classifier.misses:,} newly classified"
        )
    out: list[Classification] = []
    for result in results:
        if isinstance(result, ClassificationFailure):
            raise result.error
        out.append(result)
    return out


def build_dataset(
    classifications: Sequence[Classification],
    bars: Sequence[Bar],
    progress: Progress = SILENT,
) -> Dataset:
    """Features at each classification's knowledge time; labels from the full series."""
    ordered = sorted(classifications, key=lambda c: (c.knowledge_time, c.article_id))
    progress.stage("features", f"point-in-time features + 15m labels for {len(ordered):,} events")
    if not ordered:
        return Dataset((), {})
    clock = SimClock(ordered[0].knowledge_time)
    pit = PointInTimeBars(bars, clock)
    features: list[FeatureVector] = []
    labels: dict[EventKey, Outcome] = {}
    index = BarIndex(bars)  # built once: per-event label lookups are binary searches
    no_history = no_label = 0
    for done, classification in enumerate(ordered, 1):
        progress.tick(done, len(ordered))
        clock.advance_to(classification.knowledge_time)
        fv = build_features(classification, pit)
        if fv is None:
            no_history += 1
            continue
        label = forward_returns(fv.as_of, fv.ticker, index).get(HORIZON_MINUTES)
        if label is None:
            no_label += 1
            continue
        features.append(fv)
        labels[fv.article_id, fv.ticker, fv.as_of] = label
    progress.note(
        f"kept {len(features):,} events; dropped {no_history:,} without enough completed bars "
        f"(e.g. outside market hours), {no_label:,} without a 15m outcome"
    )
    return Dataset(tuple(features), labels)


def purge_dataset(dataset: Dataset, cut: datetime) -> Dataset:
    """Keep samples whose outcomes are fully realized before the cutoff."""
    keep = tuple(
        f
        for f in dataset.features
        if dataset.labels[f.article_id, f.ticker, f.as_of].end_time < cut
    )
    return Dataset(
        keep,
        {
            (f.article_id, f.ticker, f.as_of): dataset.labels[f.article_id, f.ticker, f.as_of]
            for f in keep
        },
    )


def evaluate_and_fit(
    dataset: Dataset, registry_path: Path, n_splits: int, progress: Progress = SILENT
) -> tuple[EvaluationReport, LinearSignalModel]:
    """Evaluate with purged folds, then refit on all eligible samples."""
    progress.stage(
        "evaluate",
        f"purged walk-forward CV: {n_splits} folds over {len(dataset.features):,} samples",
    )
    report = evaluate(
        dataset.features, dataset.labels, n_splits, registry=TrialRegistry(registry_path)
    )
    progress.note(
        f"out-of-sample days={report.n_ic_periods}, trial #{report.n_trials} in the registry"
    )
    progress.stage("fit", f"ridge model on all {len(dataset.features):,} samples")
    targets = [dataset.labels[f.article_id, f.ticker, f.as_of].raw_return for f in dataset.features]
    return report, LinearSignalModel.fit(dataset.features, targets)


async def run_offline(
    articles: Sequence[Article],
    bars: Sequence[Bar],
    classifier: Classifier,
    registry_path: Path,
    *,
    train_fraction: float = 0.5,
    n_splits: int = 4,
    initial_cash: float = 100_000.0,
    progress: Progress = SILENT,
) -> OfflineRun:
    """Research on the first part of the timeline, then replay the rest out of sample."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    classifications = await classify_articles(articles, classifier, progress=progress)
    if len(classifications) < 2:
        raise ValueError(f"need at least 2 classified events, got {len(classifications)}")
    ordered = sorted(classifications, key=lambda c: (c.knowledge_time, c.article_id))
    cut = ordered[int(len(ordered) * train_fraction)].knowledge_time
    # Purge: a training sample is usable only if its label was fully realised before replay.
    test_events = [c for c in ordered if c.knowledge_time >= cut]
    progress.stage(
        "split",
        f"at {cut:%Y-%m-%d %H:%M}Z: train on {len(ordered) - len(test_events):,} events, "
        f"replay {len(test_events):,} unseen events",
    )
    train = purge_dataset(
        build_dataset([c for c in ordered if c.knowledge_time < cut], bars, progress), cut
    )
    report, model = evaluate_and_fit(train, registry_path, n_splits, progress)
    test_bars = [b for b in bars if b.end > cut - _WARMUP]
    # Every exchange date the engine visits: bar closes plus news arriving on non-trading days.
    days = {b.end.astimezone(_EXCHANGE_TZ).date() for b in test_bars} | {
        c.knowledge_time.astimezone(_EXCHANGE_TZ).date() for c in test_events
    }
    progress.stage(
        "replay",
        f"{len(days)} days, next-open fills, costs, sizing, limits, pre-close exits",
    )
    seen = 0

    def on_session(_: object) -> None:
        nonlocal seen
        seen += 1
        progress.tick(seen, len(days))

    strategy = NewsStrategy(test_events, model, is_regular_session)
    result = await run_backtest(
        strategy, test_bars, initial_cash=initial_cash, on_session=on_session
    )
    return OfflineRun(len(classifications), report, model.model_id, cut, result)
