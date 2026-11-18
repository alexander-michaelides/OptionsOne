"""Adapter for an optionally installed local Laya router."""

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from importlib import import_module
from threading import Lock
from typing import Protocol

from pydantic import BaseModel, ValidationError

from optionsone.classify import ClassificationResponseError
from optionsone.classify._answers import DirectionWeights, ModelAnswers, Noul, Score
from optionsone.classify.schema import QUESTIONS, build_state
from optionsone.core.clock import Clock, LiveClock
from optionsone.core.types import Article, Classification, EffectType, EventType, TimeHorizon


class Router(Protocol):
    def predict(self, state: dict[str, str], questions: Mapping[str, object], /) -> object: ...


class _Choice[T](BaseModel):
    choice: T


class _Score(BaseModel):
    score: Score


class _Noul(BaseModel):
    noul: Noul


class _Direction(BaseModel):
    probabilities: DirectionWeights


class _RouterAnswers(BaseModel):
    event_type: _Choice[EventType]
    direction: _Direction
    materiality: _Score
    novelty: _Score
    certainty: _Score
    company_relevance: _Noul
    market_relevance: _Noul
    already_known: _Noul
    time_horizon: _Choice[TimeHorizon]
    effect_type: _Choice[EffectType]


class _Prediction(BaseModel):
    answers: _RouterAnswers


class LayaClassifier:
    """Run blocking router inference off-thread, one prediction per router at a time."""

    def __init__(
        self,
        router: Router,
        *,
        model_id: str = "laya",
        clock: Clock | None = None,
        latency: timedelta | None = None,
    ) -> None:
        if latency is not None and latency < timedelta(0):
            raise ValueError("fixed inference latency must be nonnegative")
        self._router = router
        self._model_id = model_id
        self._clock = LiveClock() if clock is None else clock
        self._latency = latency
        self._lock = Lock()

    @classmethod
    def from_laya(cls, *, token: str | None = None) -> LayaClassifier:
        """Load the optional Laya package and preload its default router."""
        router: Router = import_module("laya").Router(preload=True, token=token)
        return cls(router)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _predict(self, state: dict[str, str]) -> object:
        with self._lock:
            return self._router.predict(state, QUESTIONS)

    async def classify(
        self, article: Article, ticker: str, *, company_name: str | None = None
    ) -> Classification:
        state = build_state(article, ticker, company_name)
        started = self._clock.now() if self._latency is None else None
        prediction = await asyncio.to_thread(self._predict, state)
        try:
            raw = _Prediction.model_validate(prediction).answers
        except ValidationError as exc:
            raise ClassificationResponseError(
                f"invalid Laya classification for article {article.id}, ticker {ticker}"
            ) from exc
        answers = ModelAnswers(
            event_type=raw.event_type.choice,
            direction=raw.direction.probabilities,
            materiality=raw.materiality.score,
            novelty=raw.novelty.score,
            certainty=raw.certainty.score,
            company_relevance=raw.company_relevance.noul,
            market_relevance=raw.market_relevance.noul,
            already_known=raw.already_known.noul,
            time_horizon=raw.time_horizon.choice,
            effect_type=raw.effect_type.choice,
        )
        elapsed = self._clock.now() - started if started is not None else self._latency
        assert elapsed is not None
        return answers.to_classification(article, ticker, self.model_id, elapsed)
