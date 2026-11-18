"""Volatility-targeted whole-share sizing."""

from collections.abc import Sequence
from itertools import pairwise
from math import floor, isfinite, sqrt

from pydantic import BaseModel, ConfigDict, Field

from optionsone.core.types import Bar, Signal


class SizingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    target_bar_vol: float = Field(default=0.001, gt=0)
    min_bar_vol: float = Field(default=0.0001, gt=0)
    max_position_pct: float = Field(default=0.10, gt=0, le=1)
    max_shares: int = Field(default=100_000, gt=0)
    vol_window: int = Field(default=21, ge=2)


def realized_bar_vol(bars: Sequence[Bar], window: int) -> float | None:
    """EWMA root mean square of contiguous one-minute close returns."""
    recent = bars[-window:]
    variance: float | None = None
    alpha = 2 / (window + 1)
    for previous, current in pairwise(recent):
        if previous.end == current.start:
            squared = (current.close / previous.close - 1) ** 2
            variance = squared if variance is None else alpha * squared + (1 - alpha) * variance
    return None if variance is None else sqrt(variance)


def size_shares(
    signal: Signal, equity: float, price: float, bar_vol: float, config: SizingConfig
) -> int:
    """Return unsigned size using fractional per-bar volatility and the latest known price."""
    if not all(isfinite(value) for value in (equity, price, bar_vol)):
        return 0
    if equity <= 0 or price <= 0 or bar_vol < 0:
        return 0
    weight = signal.strength * config.target_bar_vol / max(bar_vol, config.min_bar_vol)
    return min(config.max_shares, floor(equity * min(weight, config.max_position_pct) / price))
