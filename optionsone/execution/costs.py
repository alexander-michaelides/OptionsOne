"""Execution commissions, half-spread and square-root market impact."""

from dataclasses import dataclass
from math import sqrt
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from optionsone.core.types import OrderSide


class Liquidity(BaseModel):
    """Point-in-time daily return volatility and average daily volume in shares."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    sigma: float = Field(ge=0)
    adv: float = Field(gt=0)


class MissingLiquidityError(ValueError):
    """Impact was enabled without a point-in-time liquidity estimate."""


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    commission: float
    spread: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.impact


class CostModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    commission_per_share: float = Field(default=0.005, ge=0)
    half_spread_bps: float = Field(default=1.0, ge=0, lt=10_000)
    eta: float = Field(default=0.0, ge=0)

    def quote(
        self, price: float, qty: int, side: OrderSide, liquidity: Liquidity | None = None
    ) -> tuple[float, ExecutionCosts]:
        impact_pct = 0.0
        if self.eta:
            if liquidity is None:
                raise MissingLiquidityError("positive eta requires sigma and ADV")
            impact_pct = self.eta * liquidity.sigma * sqrt(qty / liquidity.adv)
        costs = ExecutionCosts(
            commission=self.commission_per_share * qty,
            spread=price * qty * self.half_spread_bps / 10_000,
            impact=price * qty * impact_pct,
        )
        slippage = (costs.spread + costs.impact) / qty
        return price + (slippage if side is OrderSide.BUY else -slippage), costs


DEFAULT_COSTS: Final = CostModel()
