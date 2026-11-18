"""Point-in-time news decisions for replay and paper trading."""

from collections.abc import Callable
from datetime import datetime

from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Classification, Signal
from optionsone.features.pipeline import build_features
from optionsone.signals.model import SignalModel
from optionsone.signals.rules import passes_pre_trade


def decide(
    classification: Classification,
    pit_bars: PointInTimeBars,
    model: SignalModel,
    *,
    is_regular_session: Callable[[datetime], bool],
    min_company_relevance: float,
    min_novelty: float,
) -> Signal | None:
    """Build features, apply eligibility rules, and predict from the same PIT view."""
    fv = build_features(classification, pit_bars)
    if fv is None or not passes_pre_trade(
        fv,
        is_regular_session,
        min_company_relevance=min_company_relevance,
        min_novelty=min_novelty,
    ):
        return None
    return model.predict(fv)
