"""Forward outcomes from full research bars. Never use these as decision inputs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

from optionsone.core.types import BAR_DURATION, Bar

HORIZONS = (1, 5, 15, 30, 60)


@dataclass(frozen=True, slots=True)
class ForwardLabel:
    entry_time: datetime
    end_time: datetime
    raw_return: float
    market_adjusted_return: float | None


@dataclass(frozen=True, slots=True)
class TripleBarrierLabel:
    entry_time: datetime
    end_time: datetime
    raw_return: float
    label: Literal[-1, 0, 1]


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """Multipliers of volatility known at the decision, with a time limit in minutes."""

    volatility: float
    max_holding: int = 15
    upper_mult: float = 2.0
    lower_mult: float = 1.5


def _entry(stock: Iterable[Bar], decision: datetime, latency: timedelta) -> Bar | None:
    if latency < timedelta(0):
        raise ValueError("entry latency must be nonnegative")
    ready = decision + latency
    return min(
        (bar for bar in stock if bar.start > decision and bar.start >= ready),
        key=lambda bar: bar.start,
        default=None,
    )


def forward_returns(
    decision_time: datetime,
    symbol: str,
    bars: Sequence[Bar],
    *,
    entry_latency: timedelta = timedelta(0),
    horizons: Sequence[int] = HORIZONS,
    benchmark: str = "SPY",
) -> dict[int, ForwardLabel]:
    """Enter at the first eligible open strictly after the decision.

    Exit at the close ending exactly `horizon` clock minutes after entry.
    Missing stock exits omit that horizon. Missing aligned benchmark endpoints
    leave the adjusted return unavailable.
    """
    if any(h <= 0 for h in horizons):
        raise ValueError("horizons must be positive minutes")
    stock = {bar.start: bar for bar in bars if bar.symbol == symbol}
    entry = _entry(stock.values(), decision_time, entry_latency)
    if entry is None:
        return {}
    market = {bar.start: bar for bar in bars if bar.symbol == benchmark}
    market_entry = market.get(entry.start)
    labels: dict[int, ForwardLabel] = {}
    for horizon in horizons:
        end = entry.start + timedelta(minutes=horizon)
        exit_start = end - BAR_DURATION
        if (exit_bar := stock.get(exit_start)) is None:
            continue
        raw = exit_bar.close / entry.open - 1
        market_exit = market.get(exit_start)
        adjusted = (
            raw - (market_exit.close / market_entry.open - 1)
            if market_entry is not None and market_exit is not None
            else None
        )
        labels[horizon] = ForwardLabel(entry.start, end, raw, adjusted)
    return labels


def triple_barrier_label(
    decision_time: datetime,
    symbol: str,
    bars: Sequence[Bar],
    spec: BarrierSpec,
    *,
    entry_latency: timedelta = timedelta(0),
) -> TripleBarrierLabel | None:
    """Use decision-time volatility and check each one-minute OHLC after entry.

    An open through a barrier fills at the open. Otherwise touching both barriers
    in the same bar gives the stop priority. Timeout is class zero. Each outcome
    is known at its bar end. A missing bar censors paths still open at that point.
    """
    if spec.max_holding <= 0 or any(
        not isfinite(value) or value <= 0
        for value in (spec.volatility, spec.upper_mult, spec.lower_mult)
    ):
        raise ValueError("barrier volatility, multipliers, and holding period must be positive")
    stock = {bar.start: bar for bar in bars if bar.symbol == symbol}
    entry = _entry(stock.values(), decision_time, entry_latency)
    if entry is None:
        return None
    upper = entry.open * (1 + spec.volatility * spec.upper_mult)
    lower = entry.open * (1 - spec.volatility * spec.lower_mult)
    for offset in range(spec.max_holding):
        bar = stock.get(entry.start + offset * BAR_DURATION)
        if bar is None:
            return None
        if bar.open <= lower:
            return TripleBarrierLabel(entry.start, bar.end, bar.open / entry.open - 1, -1)
        if bar.open >= upper:
            return TripleBarrierLabel(entry.start, bar.end, bar.open / entry.open - 1, 1)
        if bar.low <= lower:
            return TripleBarrierLabel(entry.start, bar.end, lower / entry.open - 1, -1)
        if bar.high >= upper:
            return TripleBarrierLabel(entry.start, bar.end, upper / entry.open - 1, 1)
        if offset == spec.max_holding - 1:
            return TripleBarrierLabel(entry.start, bar.end, bar.close / entry.open - 1, 0)
    return None
