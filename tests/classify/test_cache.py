import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

from optionsone.classify.batcher import classify_many
from optionsone.classify.cache import CachedClassifier
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.types import Article, Classification


class _CountingClassifier(DeterministicStubClassifier):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    async def classify(self, article: Article, ticker: str) -> Classification:
        self.calls += 1
        return await super().classify(article, ticker)


async def test_second_batch_is_all_cache_hits_across_instances(
    article: Article, tmp_path: Path
) -> None:
    path = tmp_path / "classifications.sqlite3"
    first = _CountingClassifier("first-model")
    second = _CountingClassifier("first-model")
    items = [(article, "AAA"), (article, "BBB")]

    original = await classify_many(CachedClassifier(first, path), items)
    replay = await classify_many(CachedClassifier(second, path), items)

    assert first.calls == len(items)
    assert second.calls == 0
    assert all(isinstance(result, Classification) for result in replay)
    assert replay == original
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)


async def test_cache_misses_for_new_article_revision_and_model(
    article: Article, tmp_path: Path
) -> None:
    path = tmp_path / "classifications.sqlite3"
    first = _CountingClassifier("first-model")
    second = _CountingClassifier("second-model")
    revised = article.model_copy(update={"updated_at": article.updated_at + timedelta(seconds=1)})
    cache = CachedClassifier(first, path)
    await cache.classify(article, "AAA")

    await cache.classify(revised, "AAA")
    fresh_model = await CachedClassifier(second, path).classify(article, "AAA")

    assert first.calls == 2
    assert second.calls == 1
    assert fresh_model.model_id == "second-model"
