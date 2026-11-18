"""Build the canonical feature vector for a classified event."""

from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Classification, FeatureVector
from optionsone.features.market import MARKET_FEATURE_NAMES, market_features

FEATURE_NAMES = (
    "direction",
    "materiality",
    "novelty",
    "certainty",
    "company_relevance",
    "market_relevance",
    "not_already_known",
    *MARKET_FEATURE_NAMES,
)


def build_features(
    classification: Classification, bars: PointInTimeBars, benchmark: str = "SPY"
) -> FeatureVector | None:
    """Return features when the classification and required completed bars are available."""
    as_of = bars.now()
    if classification.knowledge_time > as_of:
        return None
    market = market_features(
        bars.bars_asof(classification.ticker),
        bars.bars_asof(benchmark),
        classification.knowledge_time,
    )
    if market is None:
        return None
    return FeatureVector(
        article_id=classification.article_id,
        ticker=classification.ticker,
        as_of=as_of,
        values={
            "direction": classification.direction,
            "materiality": classification.materiality,
            "novelty": classification.novelty,
            "certainty": classification.certainty,
            "company_relevance": classification.company_relevance,
            "market_relevance": classification.market_relevance,
            "not_already_known": 1 - classification.already_known,
            **market,
        },
    )
