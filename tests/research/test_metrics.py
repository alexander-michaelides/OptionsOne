from statistics import fmean

import numpy as np
import pytest

from optionsone.research.metrics import (
    deflated_sharpe_ratio,
    hac_t_stat,
    period_rank_ic,
    probabilistic_sharpe_ratio,
    rank_ic,
)


def test_rank_ic_with_ties_and_undefined_cross_sections() -> None:
    assert rank_ic([10, 10, 20, 30], [1, 2, 3, 4]) == pytest.approx(0.948683298)
    assert period_rank_ic([1, 1, 2], [2, 3, 4], [0, 0, 1]) == {}


def test_ic_distinguishes_planted_signal_from_seeded_noise() -> None:
    rng = np.random.default_rng(77)
    periods = [i for i in range(80) for _ in range(32)]
    truth = list(map(float, rng.normal(size=len(periods))))
    noise = list(map(float, rng.normal(size=len(periods))))
    scores = [x + e for x, e in zip(truth, noise, strict=True)]
    planted_ic = list(period_rank_ic(scores, truth, periods).values())
    noise_ic = list(period_rank_ic(noise, truth, periods).values())
    planted_t = hac_t_stat(planted_ic, max_lag=3)
    noise_t = hac_t_stat(noise_ic, max_lag=3)
    assert fmean(planted_ic) > 0.5
    assert planted_t is not None
    assert planted_t > 2
    assert noise_t is not None
    assert abs(noise_t) < 2


def test_hac_bartlett_weight_accounts_for_serial_correlation() -> None:
    assert hac_t_stat([1, 2, 3, 4], max_lag=1) == pytest.approx(2.5 / (1.5625 / 4) ** 0.5)


def test_sharpe_deflation_increases_with_number_of_trials() -> None:
    rng = np.random.default_rng(7)
    returns = list(map(float, rng.normal(0.15, 1.0, 120)))
    single = deflated_sharpe_ratio(returns, n_trials=1, sharpe_std=0.09)
    ten = deflated_sharpe_ratio(returns, n_trials=10, sharpe_std=0.09)
    hundred = deflated_sharpe_ratio(returns, n_trials=100, sharpe_std=0.09)
    assert single == probabilistic_sharpe_ratio(returns)
    assert single is not None
    assert ten is not None
    assert hundred is not None
    assert 0 < hundred < ten < single < 1
