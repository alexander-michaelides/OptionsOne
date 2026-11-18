"""Pre-trade eligibility independent of the exchange-calendar provider."""

from collections.abc import Callable
from datetime import datetime

from optionsone.core.types import FeatureVector


def passes_pre_trade(
    fv: FeatureVector,
    is_regular_session: Callable[[datetime], bool],
    *,
    min_company_relevance: float,
    min_novelty: float,
) -> bool:
    """Apply inclusive classification cutoffs and check the decision timestamp."""
    return (
        fv.values["company_relevance"] >= min_company_relevance
        and fv.values["novelty"] >= min_novelty
        and is_regular_session(fv.as_of)
    )
