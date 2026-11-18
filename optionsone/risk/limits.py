"""Order limits on projected signed positions, including working orders."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from optionsone.core.types import OrderIntent, OrderSide


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_position_notional_pct: float = Field(default=0.10, gt=0, le=1)
    max_gross_exposure: float = Field(default=1.0, gt=0)
    max_open_positions: int = Field(default=10, gt=0)
    max_orders_per_min: int = Field(default=30, gt=0)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reason: str


def signed_qty(order: OrderIntent) -> int:
    return order.qty if order.side is OrderSide.BUY else -order.qty


def reduces_position(order: OrderIntent, current_qty: int) -> bool:
    change = signed_qty(order)
    return change * current_qty < 0 and abs(change) <= abs(current_qty)


def pre_trade_check(
    order: OrderIntent,
    limits: RiskLimits,
    *,
    equity: float,
    quantities: Mapping[str, int],
    prices: Mapping[str, float],
    submitted_at: Iterable[datetime],
) -> RiskDecision:
    """Evaluate marked projected exposure and the trailing 60-second order count."""
    current = quantities.get(order.symbol, 0)
    if reduces_position(order, current):
        return RiskDecision(True, "reduces_position")
    if equity <= 0:
        return RiskDecision(False, "nonpositive_equity")
    cutoff = order.created_at - timedelta(minutes=1)
    if sum(cutoff < time <= order.created_at for time in submitted_at) >= limits.max_orders_per_min:
        return RiskDecision(False, "max_orders_per_min")
    projected = {**quantities, order.symbol: current + signed_qty(order)}
    notionals: dict[str, float] = {}
    for symbol, qty in projected.items():
        if qty:
            price = prices.get(symbol)
            if price is None or price <= 0:
                return RiskDecision(False, "missing_mark")
            notionals[symbol] = abs(qty) * price
    if notionals[order.symbol] > equity * limits.max_position_notional_pct:
        return RiskDecision(False, "max_position_notional_pct")
    if sum(notionals.values()) > equity * limits.max_gross_exposure:
        return RiskDecision(False, "max_gross_exposure")
    if len(notionals) > limits.max_open_positions:
        return RiskDecision(False, "max_open_positions")
    return RiskDecision(True, "within_limits")
