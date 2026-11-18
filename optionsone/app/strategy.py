"""The news strategy shared by backtest replay and (future) paper trading."""

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime

from optionsone.backtest.portfolio import PortfolioSnapshot
from optionsone.core.pit import PointInTimeBars
from optionsone.core.types import Classification, Signal
from optionsone.features.pipeline import build_features
from optionsone.signals.model import SignalModel
from optionsone.signals.rules import passes_pre_trade


class NewsStrategy:
    """Emit signals for classified articles once their knowledge time has passed."""

    def __init__(
        self,
        classifications: Sequence[Classification],
        model: SignalModel,
        is_regular_session: Callable[[datetime], bool],
        *,
        min_company_relevance: float = 0.0,
        min_novelty: float = 0.0,
    ) -> None:
        self._events = sorted(classifications, key=lambda c: (c.knowledge_time, c.article_id))
        self._model = model
        self._is_regular_session = is_regular_session
        self._min_company_relevance = min_company_relevance
        self._min_novelty = min_novelty

    def events(self) -> Iterator[tuple[datetime, Classification]]:
        return ((c.knowledge_time, c) for c in self._events)

    def on_event(
        self,
        now: datetime,
        pit_bars: PointInTimeBars,
        portfolio: PortfolioSnapshot[Classification],
    ) -> Sequence[Signal]:
        signals: list[Signal] = []
        for classification in portfolio.events:
            fv = build_features(classification, pit_bars)
            if fv is None or not passes_pre_trade(
                fv,
                self._is_regular_session,
                min_company_relevance=self._min_company_relevance,
                min_novelty=self._min_novelty,
            ):
                continue
            signal = self._model.predict(fv)
            if signal is not None:
                signals.append(signal)
        return signals
