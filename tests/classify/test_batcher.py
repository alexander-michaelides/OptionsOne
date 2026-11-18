import asyncio
from collections import Counter
from datetime import timedelta

from optionsone.classify.batcher import ClassificationFailure, classify_many
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.types import Article, Classification


async def test_batch_limits_work_dedupes_and_keeps_order_with_failures(article: Article) -> None:
    first_two_started = asyncio.Event()
    release_second = asyncio.Event()
    third_started = asyncio.Event()
    release_first = asyncio.Event()
    inflight = peak = 0
    calls: Counter[str] = Counter()
    failure = RuntimeError("provider unavailable")

    class GatedClassifier(DeterministicStubClassifier):
        async def classify(self, article: Article, ticker: str) -> Classification:
            nonlocal inflight, peak
            calls[ticker] += 1
            inflight += 1
            peak = max(peak, inflight)
            if inflight == 2:
                first_two_started.set()
            try:
                if ticker == "AAA":
                    await release_first.wait()
                if ticker == "BBB":
                    await release_second.wait()
                    raise failure
                if ticker == "CCC":
                    third_started.set()
                return await super().classify(article, ticker)
            finally:
                inflight -= 1

    items = [(article, ticker) for ticker in ("AAA", "BBB", "AAA", "CCC", "BBB")]
    async with asyncio.timeout(2), asyncio.TaskGroup() as tasks:
        batch = tasks.create_task(classify_many(GatedClassifier(), items, concurrency=2))
        await first_two_started.wait()
        assert not third_started.is_set()
        release_second.set()
        await third_started.wait()
        release_first.set()

    outcomes = batch.result()
    assert peak == 2
    assert calls == {"AAA": 1, "BBB": 1, "CCC": 1}
    assert [outcome.ticker for outcome in outcomes] == ["AAA", "BBB", "AAA", "CCC", "BBB"]
    first, failed, duplicate, third, failed_duplicate = outcomes
    assert isinstance(first, Classification)
    assert isinstance(third, Classification)
    assert duplicate == first
    assert isinstance(failed, ClassificationFailure)
    assert (failed.article, failed.error) == (article, failure)
    assert failed_duplicate == failed


async def test_batch_does_not_dedupe_different_revisions(article: Article) -> None:
    revised = article.model_copy(
        update={
            "updated_at": article.updated_at + timedelta(minutes=1),
            "knowledge_time": article.knowledge_time + timedelta(minutes=1),
        }
    )

    before, after = await classify_many(
        DeterministicStubClassifier(), [(article, "AAA"), (revised, "AAA")]
    )

    assert isinstance(before, Classification)
    assert isinstance(after, Classification)
    assert (before.knowledge_time, after.knowledge_time) == (
        article.knowledge_time,
        revised.knowledge_time,
    )
