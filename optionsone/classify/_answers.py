"""Validation and domain mapping for model answers."""

from datetime import timedelta
from typing import Annotated, Self

from pydantic import Field, FiniteFloat, model_validator

from optionsone.classify.schema import PROMPT_VERSION
from optionsone.core.types import (
    Article,
    Classification,
    EffectType,
    EventType,
    Frozen,
    TimeHorizon,
)

type Score = Annotated[int, Field(strict=True, ge=0, le=4)]
type Noul = Annotated[float, Field(strict=True, ge=0, le=1)]
type Weight = Annotated[FiniteFloat, Field(strict=True, ge=0)]


class DirectionWeights(Frozen):
    positive: Weight
    neutral: Weight
    negative: Weight

    @model_validator(mode="after")
    def _nonzero(self) -> Self:
        if max(self.positive, self.neutral, self.negative) == 0:
            raise ValueError("direction weights must have positive mass")
        return self

    def probabilities(self) -> tuple[float, float, float]:
        peak = max(self.positive, self.neutral, self.negative)
        positive, neutral, negative = (
            self.positive / peak,
            self.neutral / peak,
            self.negative / peak,
        )
        total = positive + neutral + negative
        return positive / total, neutral / total, negative / total


class ModelAnswers(Frozen):
    event_type: EventType
    direction: DirectionWeights
    materiality: Score
    novelty: Score
    certainty: Score
    company_relevance: Noul
    market_relevance: Noul
    already_known: Noul
    time_horizon: TimeHorizon
    effect_type: EffectType

    def to_classification(
        self, article: Article, ticker: str, model_id: str, latency: timedelta
    ) -> Classification:
        positive, neutral, negative = self.direction.probabilities()
        return Classification(
            article_id=article.id,
            ticker=ticker,
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            event_type=self.event_type,
            effect_type=self.effect_type,
            time_horizon=self.time_horizon,
            p_positive=positive,
            p_neutral=neutral,
            p_negative=negative,
            materiality=self.materiality / 4,
            novelty=self.novelty / 4,
            certainty=self.certainty / 4,
            company_relevance=self.company_relevance,
            market_relevance=self.market_relevance,
            already_known=self.already_known,
            knowledge_time=article.knowledge_time + latency,
        )
