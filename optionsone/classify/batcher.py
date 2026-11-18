"""Bounded concurrent classification with ordered per-item results."""

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from optionsone.core.ports import Classifier
from optionsone.core.types import Article, Classification


@dataclass(frozen=True, slots=True)
class ClassificationFailure:
    article: Article
    ticker: str
    error: Exception


type ClassificationResult = Classification | ClassificationFailure


async def classify_many(
    classifier: Classifier,
    items: Iterable[tuple[Article, str]],
    concurrency: int = 8,
    on_done: Callable[[int, int], None] | None = None,
) -> list[ClassificationResult]:
    """Return one result per input, sharing work for matching article revisions and tickers."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    limit = asyncio.Semaphore(concurrency)

    finished = 0
    total = 0

    async def classify_one(article: Article, ticker: str) -> ClassificationResult:
        nonlocal finished
        async with limit:
            try:
                return await classifier.classify(article, ticker)
            except Exception as exc:
                return ClassificationFailure(article, ticker, exc)
            finally:
                finished += 1
                if on_done is not None:
                    on_done(finished, total)

    unique: dict[tuple[int, datetime, str], asyncio.Task[ClassificationResult]] = {}
    ordered: list[asyncio.Task[ClassificationResult]] = []
    pairs = list(items)
    total = len({(article.id, article.updated_at, ticker) for article, ticker in pairs})
    async with asyncio.TaskGroup() as group:
        for article, ticker in pairs:
            key = (article.id, article.updated_at, ticker)
            task = unique.get(key)
            if task is None:
                task = group.create_task(classify_one(article, ticker))
                unique[key] = task
            ordered.append(task)
    return [task.result() for task in ordered]
