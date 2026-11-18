"""Immutable domain models shared by every layer.

Every record that can feed a decision carries a `knowledge_time`: the earliest
instant the system could have known it. Decisions may only use records whose
knowledge_time is <= the clock's now().
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]

BAR_DURATION = timedelta(minutes=1)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Bar(Frozen):
    """One-minute OHLCV bar. `start` is the open of the interval; data is known at `end`."""

    symbol: str
    start: UtcDatetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    vwap: float | None = None

    @property
    def end(self) -> datetime:
        return self.start + BAR_DURATION

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> Bar:
        if not (self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high):
            raise ValueError(f"inconsistent OHLC for {self.symbol} at {self.start}")
        return self


class Article(Frozen):
    """A specific revision of a news article."""

    id: int
    headline: str
    summary: str = ""
    content: str = ""
    source: str = ""
    symbols: tuple[str, ...]
    published_at: UtcDatetime
    updated_at: UtcDatetime
    knowledge_time: UtcDatetime

    @model_validator(mode="after")
    def _knowledge_after_publish(self) -> Article:
        if self.knowledge_time < self.published_at:
            raise ValueError("knowledge_time cannot precede published_at")
        return self


class EventType(StrEnum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ANALYST = "analyst"
    CONTRACT = "contract"
    CAPEX = "capex"
    PRODUCT = "product"
    MA = "ma"
    PARTNERSHIP = "partnership"
    LEGAL = "legal"
    REGULATORY = "regulatory"
    MANAGEMENT = "management"
    FINANCING = "financing"
    OPERATIONS = "operations"
    MACRO = "macro"
    OTHER = "other"


class EffectType(StrEnum):
    REVENUE = "revenue"
    COST = "cost"
    CAPITAL = "capital"
    OPERATIONS = "operations"
    COMPETITION = "competition"
    RISK = "risk"
    EXPECTATIONS = "expectations"
    NONE = "none"


class TimeHorizon(StrEnum):
    IMMEDIATE = "immediate"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Classification(Frozen):
    """LLM answers for one (article revision, ticker), preserving the legacy 10-question schema."""

    article_id: int
    ticker: str
    model_id: str
    prompt_version: str
    event_type: EventType
    effect_type: EffectType
    time_horizon: TimeHorizon
    p_positive: Probability
    p_neutral: Probability
    p_negative: Probability
    materiality: Unit
    novelty: Unit
    certainty: Unit
    company_relevance: Unit
    market_relevance: Unit
    already_known: Unit
    knowledge_time: UtcDatetime

    @property
    def direction(self) -> float:
        """Signed expected direction in [-1, 1]."""
        return self.p_positive - self.p_negative

    @model_validator(mode="after")
    def _probabilities_sum(self) -> Classification:
        total = self.p_positive + self.p_neutral + self.p_negative
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"direction probabilities sum to {total}, expected 1")
        return self


class FeatureVector(Frozen):
    article_id: int
    ticker: str
    as_of: UtcDatetime
    values: dict[str, float]


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


class Signal(Frozen):
    article_id: int
    ticker: str
    as_of: UtcDatetime
    side: Side
    strength: Unit
    horizon: timedelta
    model_id: str


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderIntent(Frozen):
    """A market order request. `client_order_id` makes submission idempotent."""

    client_order_id: str
    symbol: str
    side: OrderSide
    qty: int = Field(gt=0)
    created_at: UtcDatetime
    reason: str = ""


class Fill(Frozen):
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: int = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(ge=0)
    filled_at: UtcDatetime


class Position(Frozen):
    symbol: str
    qty: int
    avg_price: float
