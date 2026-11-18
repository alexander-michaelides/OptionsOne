from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from anyio.to_thread import run_sync
from pydantic import JsonValue, TypeAdapter

from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.offline import run_offline
from optionsone.app.research_cmd import (
    DayProgress,
    EmptyStoreError,
    backfill,
    backtest_store,
    research,
)
from optionsone.classify.cache import CachedClassifier
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.clock import SimClock
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


async def test_store_backtest_replays_backfilled_data_like_the_in_memory_pipeline(
    tmp_path: Path,
) -> None:
    data = synthetic_fixture(days=10, seed=7)
    start, end = data.bars[0].start, data.bars[-1].end
    store_root = tmp_path / "store"
    await backfill(
        store_root, (), start, end, news=FakeNews(data.articles), bars=FakeBars(data.bars)
    )

    stored = await backtest_store(
        store_root, start, end, DeterministicStubClassifier(), registry_path=tmp_path / "a.jsonl"
    )
    direct = await run_offline(
        data.articles, data.bars, DeterministicStubClassifier(), tmp_path / "b.jsonl"
    )
    assert stored.backtest.report.trades
    assert stored.backtest.signals >= len(stored.backtest.report.trades)
    assert stored.model_id == direct.model_id
    assert stored.replay_start == direct.replay_start
    assert stored.backtest.report.net_return == direct.backtest.report.net_return


async def test_store_backtest_universe_narrows_events_and_empty_window_is_an_error(
    tmp_path: Path,
) -> None:
    data = synthetic_fixture(days=6, seed=3)
    start, end = data.bars[0].start, data.bars[-1].end
    store_root = tmp_path / "store"
    await backfill(
        store_root, (), start, end, news=FakeNews(data.articles), bars=FakeBars(data.bars)
    )
    only_aaa = await backtest_store(
        store_root,
        start,
        end,
        DeterministicStubClassifier(),
        registry_path=tmp_path / "t.jsonl",
        symbols=("aaa",),
    )
    assert only_aaa.n_classifications == sum("AAA" in a.symbols for a in data.articles)
    assert {trade.symbol for trade in only_aaa.backtest.report.trades} <= {"AAA"}

    with pytest.raises(EmptyStoreError):
        await backtest_store(
            store_root,
            end + timedelta(days=30),
            end + timedelta(days=31),
            DeterministicStubClassifier(),
            registry_path=tmp_path / "t.jsonl",
        )


async def test_small_account_skips_are_reported_not_silent(tmp_path: Path) -> None:
    data = synthetic_fixture(days=10, seed=7)  # fixture prices are about $100-160
    run = await run_offline(
        data.articles,
        data.bars,
        DeterministicStubClassifier(),
        tmp_path / "t.jsonl",
        initial_cash=500.0,
    )
    skipped = dict(run.backtest.skipped)
    assert not run.backtest.report.trades
    assert skipped["size_below_one_share"] > 0
    assert sum(skipped.values()) == run.backtest.signals


class FlakyNews:
    """Serves the fixture but fails on one day, like a dropped connection mid-run."""

    def __init__(self, articles: Sequence[Article], fail_day: datetime | None) -> None:
        self.articles, self.fail_day, self.calls = articles, fail_day, 0

    async def fetch(self, start: datetime, end: datetime) -> Sequence[Article]:
        self.calls += 1
        if self.fail_day is not None and start <= self.fail_day < end:
            raise ConnectionError("simulated network drop")
        return [a for a in self.articles if start <= a.published_at < end]


async def test_interrupted_backfill_resumes_from_the_failed_day(tmp_path: Path) -> None:
    data = synthetic_fixture(days=4)
    start = data.bars[0].start.replace(hour=0, minute=0)
    end = data.bars[-1].start.replace(hour=0, minute=0) + timedelta(days=1)
    third_day = sorted({b.start.date() for b in data.bars})[2]
    fail_at = datetime.combine(third_day, datetime.min.time(), tzinfo=start.tzinfo)
    bars = FakeBars(data.bars)

    with pytest.raises(ConnectionError):
        await backfill(tmp_path, (), start, end, news=FlakyNews(data.articles, fail_at), bars=bars)

    days: list[DayProgress] = []
    news = FlakyNews(data.articles, None)
    report = await backfill(tmp_path, (), start, end, news=news, bars=bars, on_day=days.append)
    resumed = [d.day for d in days if d.resumed]
    assert resumed  # completed days were skipped
    assert max(resumed) < third_day
    assert news.calls == len(days) - len(resumed)  # only unfinished days were fetched
    stored = ParquetStore(tmp_path).load_bars(sorted({b.symbol for b in data.bars}), start, end)
    assert len(stored) == len(data.bars)  # nothing lost, nothing duplicated
    assert report.bars_added == len(data.bars) - sum(
        1 for b in data.bars if b.start.date() < third_day
    )


async def test_open_window_is_not_marked_done_and_universes_are_separate(tmp_path: Path) -> None:
    data = synthetic_fixture(days=2)
    start = data.bars[0].start.replace(hour=0, minute=0)
    end = start + timedelta(days=1)
    bars = FakeBars(data.bars)
    during_day = SimClock(start + timedelta(hours=15))
    days: list[DayProgress] = []
    await backfill(
        tmp_path,
        (),
        start,
        end,
        news=FakeNews(data.articles),
        bars=bars,
        clock=during_day,
        on_day=days.append,
    )
    await backfill(
        tmp_path, (), start, end, news=FakeNews(data.articles), bars=bars, on_day=days.append
    )
    assert [d.resumed for d in days] == [False, False]  # still-open day was refetched

    await backfill(
        tmp_path, ("AAA",), start, end, news=FakeNews(data.articles), bars=bars, on_day=days.append
    )
    await backfill(
        tmp_path, (), start, end, news=FakeNews(data.articles), bars=bars, on_day=days.append
    )
    assert [d.resumed for d in days[2:]] == [False, True]
