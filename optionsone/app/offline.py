"""Offline research + backtest pipeline: classify, build PIT features, evaluate, replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from optionsone.app.strategy import NewsStrategy
from optionsone.backtest.engine import BacktestResult, run_backtest
from optionsone.classify.batcher import ClassificationFailure, classify_many
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
    articles: Sequence[Article], classifier: Classifier, concurrency: int = 8
) -> list[Classification]:
    items = [(article, ticker) for article in articles for ticker in article.symbols]
    results = await classify_many(classifier, items, concurrency=concurrency)
    out: list[Classification] = []
    for result in results:
        if isinstance(result, ClassificationFailure):
            raise result.error
        out.append(result)
    return out


def build_dataset(classifications: Sequence[Classification], bars: Sequence[Bar]) -> Dataset:
    """Features at each classification's knowledge time; labels from the full series."""
    ordered = sorted(classifications, key=lambda c: (c.knowledge_time, c.article_id))
    if not ordered:
        return Dataset((), {})
    clock = SimClock(ordered[0].knowledge_time)
    pit = PointInTimeBars(bars, clock)
    features: list[FeatureVector] = []
    labels: dict[EventKey, Outcome] = {}
    index = BarIndex(bars)  # built once: per-event label lookups are binary searches
    for classification in ordered:
        clock.advance_to(classification.knowledge_time)
        fv = build_features(classification, pit)
        if fv is None:
            continue
        label = forward_returns(fv.as_of, fv.ticker, index).get(HORIZON_MINUTES)
        if label is None:
            continue
        features.append(fv)
        labels[fv.article_id, fv.ticker, fv.as_of] = label
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
    dataset: Dataset, registry_path: Path, n_splits: int
) -> tuple[EvaluationReport, LinearSignalModel]:
    """Evaluate with purged folds, then refit on all eligible samples."""
    report = evaluate(
        dataset.features, dataset.labels, n_splits, registry=TrialRegistry(registry_path)
    )
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
) -> OfflineRun:
    """Research on the first part of the timeline, then replay the rest out of sample."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    classifications = await classify_articles(articles, classifier)
    if len(classifications) < 2:
        raise ValueError(f"need at least 2 classified events, got {len(classifications)}")
    ordered = sorted(classifications, key=lambda c: (c.knowledge_time, c.article_id))
    cut = ordered[int(len(ordered) * train_fraction)].knowledge_time
    # Purge: a training sample is usable only if its label was fully realised before replay.
    train = purge_dataset(build_dataset([c for c in ordered if c.knowledge_time < cut], bars), cut)
    report, model = evaluate_and_fit(train, registry_path, n_splits)
    test_events = [c for c in ordered if c.knowledge_time >= cut]
    test_bars = [b for b in bars if b.end > cut - _WARMUP]
    strategy = NewsStrategy(test_events, model, is_regular_session)
    result = await run_backtest(strategy, test_bars, initial_cash=initial_cash)
    return OfflineRun(len(classifications), report, model.model_id, cut, result)
