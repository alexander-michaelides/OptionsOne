"""Bounded concurrent classification with ordered per-item results."""

import asyncio
from collections.abc import Iterable
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
) -> list[ClassificationResult]:
    """Return one result per input, sharing work for matching article revisions and tickers."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    limit = asyncio.Semaphore(concurrency)

    async def classify_one(article: Article, ticker: str) -> ClassificationResult:
        async with limit:
            try:
                return await classifier.classify(article, ticker)
            except Exception as exc:
                return ClassificationFailure(article, ticker, exc)

    unique: dict[tuple[int, datetime, str], asyncio.Task[ClassificationResult]] = {}
    ordered: list[asyncio.Task[ClassificationResult]] = []
    async with asyncio.TaskGroup() as group:
        for article, ticker in items:
            key = (article.id, article.updated_at, ticker)
            task = unique.get(key)
            if task is None:
                task = group.create_task(classify_one(article, ticker))
                unique[key] = task
            ordered.append(task)
    return [task.result() for task in ordered]
