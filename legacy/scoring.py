# handles the scoring of each signal

import math
from dataclasses import dataclass

from news import NewsSignal
from market import MarketContext

@dataclass
class ScoreWeights:
    direction: float = 0.20
    materiality: float = 0.15
    novelty: float = 0.12
    certainty: float = 0.08
    company_relevance: float = 0.07
    market_relevance: float = 0.07
    already_known: float = 0.10

    freshness: float = 0.08
    relative_volume: float = 0.07
    alignment: float = 0.06

DEFAULT_WEIGHTS = ScoreWeights()

def _freshness(
    return_since_news: float,
) -> float:

    move = abs(return_since_news)

    if move < 0.0025:
        return 1.00

    if move < 0.005:
        return 0.90

    if move < 0.010:
        return 0.70

    if move < 0.020:
        return 0.40

    return 0.15


def _volume_quality(
    relative_volume: float,
) -> float:

    if relative_volume >= 2.0:
        return 1.00

    if relative_volume >= 1.5:
        return 0.90

    if relative_volume >= 1.0:
        return 0.75

    if relative_volume >= 0.5:
        return 0.50

    return 0.25


def _alignment(
    direction: float,
    return_5m: float | None,
) -> float:

    if (
        return_5m is None
        or not math.isfinite(return_5m)
    ):
        return 0.50

    if direction * return_5m > 0:
        return 1.00

    if abs(return_5m) < 0.001:
        return 0.70

    return 0.30

@dataclass
class ScoreFeatures:
    direction: float
    materiality: float
    novelty: float
    certainty: float
    company_relevance: float
    market_relevance: float
    already_known: float

    freshness: float
    relative_volume: float
    alignment: float

def get_features(
    signal: NewsSignal,
    ctx: MarketContext,
) -> ScoreFeatures:

    return ScoreFeatures(
        direction=abs(signal.direction),

        materiality=signal.materiality,
        novelty=signal.novelty,
        certainty=signal.certainty,

        company_relevance=
            signal.company_relevance,

        market_relevance=
            signal.market_relevance,

        # Flip this so higher always means "better".
        already_known=
            1.0 - signal.already_known,

        freshness=_freshness(
            ctx.return_since_news
        ),

        relative_volume=_volume_quality(
            ctx.relative_volume
        ),

        alignment=_alignment(
            signal.direction,
            ctx.return_5m,
        ),
    )

def score(
    signal: NewsSignal,
    ctx: MarketContext,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> float:

    f = get_features(
        signal,
        ctx,
    )
    quality = (
        weights.direction
        * f.direction

        + weights.materiality
        * f.materiality

        + weights.novelty
        * f.novelty

        + weights.certainty
        * f.certainty

        + weights.company_relevance
        * f.company_relevance

        + weights.market_relevance
        * f.market_relevance

        + weights.already_known
        * f.already_known

        + weights.freshness
        * f.freshness

        + weights.relative_volume
        * f.relative_volume

        + weights.alignment
        * f.alignment
    )

    return max(
        0.0,
        min(1.0, quality),
    )