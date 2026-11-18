"""Period rank correlations and non-normal Sharpe inference."""

from collections.abc import Hashable, Sequence
from itertools import groupby
from math import e, sqrt
from statistics import NormalDist, fmean, stdev

import numpy as np
from numpy.typing import NDArray

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def _ranks(values: Sequence[float]) -> NDArray[np.float64]:
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    order = sorted(range(len(values)), key=values.__getitem__)
    for _, indices in groupby(order, key=values.__getitem__):
        tied = list(indices)
        ranks[tied] = position + (1 + len(tied)) / 2
        position += len(tied)
    return ranks


def rank_ic(predictions: Sequence[float], targets: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties, or None if undefined."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must align")
    if len(predictions) < 2 or len(set(predictions)) == 1 or len(set(targets)) == 1:
        return None
    a, b = _ranks(predictions), _ranks(targets)
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / sqrt(float(a @ a) * float(b @ b)))


def period_rank_ic[P: Hashable](
    predictions: Sequence[float], targets: Sequence[float], periods: Sequence[P]
) -> dict[P, float]:
    """Compute one cross-sectional IC per period, omitting undefined periods."""
    grouped: dict[P, tuple[list[float], list[float]]] = {}
    for prediction, target, period in zip(predictions, targets, periods, strict=True):
        scores, outcomes = grouped.setdefault(period, ([], []))
        scores.append(prediction)
        outcomes.append(target)
    return {
        period: ic
        for period, (scores, outcomes) in grouped.items()
        if (ic := rank_ic(scores, outcomes)) is not None
    }


def hac_t_stat(values: Sequence[float], max_lag: int = 1) -> float | None:
    """Newey-West t-statistic of a mean with Bartlett weights and ordered periods."""
    if max_lag < 0:
        raise ValueError("HAC lag must be nonnegative")
    n = len(values)
    if n < 2 or len(set(values)) == 1:
        return None
    series = np.array(values, dtype=np.float64)
    mean = float(series.mean())
    resid = series - mean
    lag_count = min(n - 1, max_lag)
    long_run = float(resid @ resid) / n
    for lag in range(1, lag_count + 1):
        weight = 1 - lag / (lag_count + 1)
        long_run += 2 * weight * (float(resid[lag:] @ resid[:-lag]) / n)
    return mean / sqrt(long_run / n) if long_run > 0 else None


def hit_rate(signed_returns: Sequence[float]) -> float | None:
    """Fraction of executed trades with strictly positive gross signed returns."""
    return fmean(value > 0 for value in signed_returns) if len(signed_returns) else None


def sharpe_ratio(returns: Sequence[float]) -> float | None:
    """Unannualized per-period Sharpe using the sample standard deviation."""
    if len(returns) < 2 or (volatility := stdev(returns)) == 0:
        return None
    return fmean(returns) / volatility


def probabilistic_sharpe_ratio(
    returns: Sequence[float], benchmark_sharpe: float = 0.0
) -> float | None:
    """Probability of beating a per-period Sharpe benchmark using Pearson kurtosis."""
    if len(returns) < 3 or (observed := sharpe_ratio(returns)) is None:
        return None
    values = np.array(returns, dtype=np.float64)
    centered = (values - values.mean()) / values.std()
    skew = float(np.mean(centered**3))
    kurtosis = float(np.mean(centered**4))
    variance = 1 - skew * observed + (kurtosis - 1) * observed**2 / 4
    if variance <= 0:
        return None
    return _NORMAL.cdf((observed - benchmark_sharpe) * sqrt((len(returns) - 1) / variance))


def deflated_sharpe_ratio(
    returns: Sequence[float], n_trials: int, sharpe_std: float
) -> float | None:
    """Bailey and Lopez de Prado's PSR against the expected best null trial Sharpe."""
    if n_trials < 1 or not np.isfinite(sharpe_std) or sharpe_std < 0:
        raise ValueError(
            "trial count must be positive and Sharpe dispersion finite and nonnegative"
        )
    if n_trials == 1:
        return probabilistic_sharpe_ratio(returns)
    # Symmetry avoids subtracting very small tail probabilities from one.
    benchmark = -sharpe_std * (
        (1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 / n_trials)
        + _EULER_GAMMA * _NORMAL.inv_cdf(1 / (n_trials * e))
    )
    return probabilistic_sharpe_ratio(returns, benchmark)
