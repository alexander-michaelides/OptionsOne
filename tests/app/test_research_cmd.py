from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from anyio.to_thread import run_sync
from pydantic import JsonValue, TypeAdapter

from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.research_cmd import backfill, research
from optionsone.classify.cache import CachedClassifier
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.types import Article, Bar, Classification
from optionsone.ingest.store import ParquetStore
from optionsone.ingest.validate import BarTimestamp
from optionsone.signals.model import LinearSignalModel


@dataclass(frozen=True, slots=True)
class FakeNews:
    articles: Sequence[Article]

    async def fetch(self, start: datetime, end: datetime) -> Sequence[Article]:
        return [article for article in self.articles if start <= article.published_at <= end]


@dataclass(frozen=True, slots=True)
class FakeBars:
    bars: Sequence[Bar]

    async def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> Sequence[Bar]:
        return [bar for bar in self.bars if bar.symbol in symbols and start <= bar.start <= end]


class CountingStub(DeterministicStubClassifier):
    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, article: Article, ticker: str) -> Classification:
        self.calls += 1
        return await super().classify(article, ticker)


@pytest.mark.parametrize("universe", [(), ("AAA", "UNMENTIONED")])
async def test_backfill_filters_news_and_bars_and_reports_source_duplicates(
    tmp_path: Path, universe: tuple[str, ...]
) -> None:
    data = synthetic_fixture(days=1)
    aaa = next(article for article in data.articles if article.symbols == ("AAA",))
    mixed = aaa.model_copy(update={"symbols": ("AAA", "BBB")})
    bbb = next(article for article in data.articles if article.symbols == ("BBB",))
    first, last = data.bars[0], max(data.bars, key=lambda bar: bar.start)
    close = last.end
    start, end = first.start - timedelta(hours=1), close + timedelta(hours=1)
    after_close = first.model_copy(update={"start": close})
    out_of_universe = first.model_copy(update={"symbol": "UNMENTIONED"})
    report = await backfill(
        tmp_path,
        universe,
        start,
        end,
        news=FakeNews((mixed, bbb)),
        bars=FakeBars((*data.bars, first, after_close, out_of_universe)),
    )

    store = ParquetStore(tmp_path)
    expected_symbols = ("SPY", "AAA") if universe else ("SPY", "AAA", "BBB")
    stored = await run_sync(store.load_bars, ("SPY", "AAA", "BBB", "UNMENTIONED"), start, end)
    assert {(bar.symbol, bar.start) for bar in stored} == {
        (bar.symbol, bar.start) for bar in data.bars if bar.symbol in expected_symbols
    }
    assert (report.articles_fetched, report.articles_added) == ((1, 1) if universe else (2, 2))
    assert report.bars_added == len(stored)
    assert report.bars_fetched == len(stored) + 1
    assert report.validation.duplicates == (BarTimestamp(first.symbol, first.start),)


async def test_backfilled_research_exports_model_and_reuses_classifications(tmp_path: Path) -> None:
    data = synthetic_fixture(days=4)
    start, end = data.bars[0].start, data.bars[-1].end
    store_root = tmp_path / "store"
    news, bars = FakeNews(data.articles), FakeBars(data.bars)
    first = await backfill(store_root, (), start, end, news=news, bars=bars)
    second = await backfill(store_root, (), start, end, news=news, bars=bars)
    assert (first.articles_added, first.bars_added) == (len(data.articles), len(data.bars))
    assert (second.articles_added, second.bars_added) == (0, 0)

    model_out = tmp_path / "artifacts" / "model.json"
    report_out = tmp_path / "artifacts" / "report.json"
    registry_path = tmp_path / "state" / "trials.jsonl"
    classifier = CountingStub()
    cache = tmp_path / "classifications.sqlite"
    result = await research(
        store_root,
        start,
        end,
        CachedClassifier(classifier, cache),
        model_out=model_out,
        report_out=report_out,
        registry_path=registry_path,
    )
    assert classifier.calls == len(data.articles)
    loaded = LinearSignalModel.from_json(await AsyncPath(model_out).read_text(encoding="utf-8"))
    assert loaded.model_id == result.model_id
    saved = TypeAdapter(dict[str, JsonValue]).validate_json(
        await AsyncPath(report_out).read_text(encoding="utf-8")
    )
    assert saved == {**asdict(result.evaluation), "passes_promotion_gate": False}
    assert result.passes_promotion_gate is False  # Four days cannot meet the promotion minimum.
    assert result.evaluation.n_oos > 0

    repeated = await research(
        store_root,
        start,
        end,
        CachedClassifier(classifier, cache),
        model_out=model_out,
        report_out=report_out,
        registry_path=registry_path,
    )
    assert classifier.calls == len(data.articles)
    assert repeated.model_id == result.model_id
    assert repeated.evaluation.n_trials == 2
