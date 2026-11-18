"""Deterministic classifier for offline replay and tests."""

import hashlib
import json
import re
from datetime import timedelta
from typing import Final

from optionsone.classify._answers import DirectionWeights, ModelAnswers
from optionsone.core.types import Article, Classification, EffectType, EventType, TimeHorizon

_POSITIVE: Final = frozenset({"beats", "raises", "upgrade", "upgrades"})
_NEGATIVE: Final = frozenset({"misses", "cuts", "downgrade", "downgrades", "lawsuit"})


class DeterministicStubClassifier:
    @property
    def model_id(self) -> str:
        return "deterministic-stub"

    async def classify(self, article: Article, ticker: str) -> Classification:
        seed = json.dumps((article.id, ticker, article.headline)).encode()
        digest = hashlib.sha256(seed).digest()
        words = set(re.findall(r"\b[a-z]+\b", article.headline.lower()))
        polarity = len(words & _POSITIVE) - len(words & _NEGATIVE)
        events, effects, horizons = tuple(EventType), tuple(EffectType), tuple(TimeHorizon)
        answers = ModelAnswers(
            event_type=events[digest[0] % len(events)],
            direction=DirectionWeights(
                positive=1 + digest[1] / 255 + 2 * (polarity > 0),
                neutral=1 + digest[2] / 255 + 2 * (polarity == 0),
                negative=1 + digest[3] / 255 + 2 * (polarity < 0),
            ),
            materiality=digest[4] % 5,
            novelty=digest[5] % 5,
            certainty=digest[6] % 5,
            company_relevance=digest[7] / 255,
            market_relevance=digest[8] / 255,
            already_known=digest[9] / 255,
            time_horizon=horizons[digest[10] % len(horizons)],
            effect_type=effects[digest[11] % len(effects)],
        )
        return answers.to_classification(article, ticker, self.model_id, timedelta(0))
