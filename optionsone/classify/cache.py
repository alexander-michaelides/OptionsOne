"""Persistent cache for successful classifications of article revisions."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from optionsone.classify.schema import PROMPT_VERSION
from optionsone.core.ports import Classifier
from optionsone.core.types import Article, Classification

type _CacheKey = tuple[int, str, str, str, str]


class CachedClassifier:
    """Decorate a classifier with a file-backed SQLite WAL cache."""

    def __init__(self, classifier: Classifier, path: str | Path) -> None:
        self._classifier = classifier
        self._path = path
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._classifier.model_id

    def _initialize(self) -> None:
        with closing(sqlite3.connect(self._path, autocommit=True)) as db:
            db.execute("PRAGMA journal_mode=WAL").fetchone()
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS classifications (
                    article_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (article_id, updated_at, ticker, model_id, prompt_version)
                )
                """
            )

    def _lookup(self, key: _CacheKey) -> Classification | None:
        with closing(sqlite3.connect(self._path)) as db:
            row: tuple[str] | None = db.execute(
                """
                SELECT payload FROM classifications
                WHERE (article_id, updated_at, ticker, model_id, prompt_version) = (?, ?, ?, ?, ?)
                """,
                key,
            ).fetchone()
        return None if row is None else Classification.model_validate_json(row[0])

    def _store(self, key: _CacheKey, classification: Classification) -> None:
        with closing(sqlite3.connect(self._path, autocommit=True)) as db:
            db.execute(
                """
                INSERT INTO classifications
                    (article_id, updated_at, ticker, model_id, prompt_version, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (*key, classification.model_dump_json()),
            )

    async def classify(self, article: Article, ticker: str) -> Classification:
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    await asyncio.to_thread(self._initialize)
                    self._initialized = True

        key = (article.id, article.updated_at.isoformat(), ticker, self.model_id, PROMPT_VERSION)
        cached = await asyncio.to_thread(self._lookup, key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        classification = await self._classifier.classify(article, ticker)
        await asyncio.to_thread(self._store, key, classification)
        return classification
