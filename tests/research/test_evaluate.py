from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from optionsone.core.types import FeatureVector
from optionsone.research.evaluate import (
    EvaluationReport,
    EvaluationSpec,
    EventKey,
    Outcome,
    evaluate,
    passes_promotion_gate,
)
from optionsone.research.labels import ForwardLabel
from optionsone.research.registry import TrialRegistry
from optionsone.signals.model import RidgeSpec


def test_evaluation_joins_events_logs_fits_and_deflates_against_all_attempts(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(52)
    start = datetime(2025, 1, 2, 15, tzinfo=UTC)
    rows: list[FeatureVector] = []
    labels: dict[EventKey, Outcome] = {}
    for day in range(15):
        now = start + timedelta(days=day)
        entry = now + timedelta(minutes=1)
        for idx, direction in enumerate(list(map(float, rng.normal(size=16)))):
            ticker = f"SYMBOL{idx}"
            rows.append(
                FeatureVector(
                    ticker=ticker, article_id=day, as_of=now, values={"direction": direction}
                )
            )
            # One unlabeled row per day, including in the middle of ticker order.
            if idx != 4:
                raw_return = 0.003 * direction + float(rng.normal(0, 0.002))
                labels[(day, ticker, now)] = ForwardLabel(
                    entry, entry + timedelta(minutes=15), raw_return, None
                )
    spec = EvaluationSpec(ridge=RidgeSpec(feature_names=("direction",)), cost_bps=5)
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="distinct event times"):
        evaluate(rows, labels, 15, registry=registry, spec=spec)
    report = evaluate(list(reversed(rows)), labels, 4, registry=registry, spec=spec)
    records = registry.records()
    starts = [r for r in records if r.kind == "trial_started"]
    fits = [r for r in records if r.kind == "fit_completed"]
    assert len(starts) == report.n_trials == 2
    assert starts[0].data_hash == starts[1].data_hash
    assert len(fits) == 4
    assert len({r.data_hash for r in fits}) == 4
    assert all(r.params["feature_names"] == ("direction",) and r.model_id for r in fits)
    assert report.n_oos == report.n_trades == 180
    assert report.n_ic_periods == 12
    assert report.oos_ic_mean is not None
    assert report.oos_ic_mean > 0.65
    assert report.ic_t_stat is not None
    assert report.ic_t_stat > 2
    assert report.hit_rate is not None
    assert report.hit_rate > 0.7
    assert report.gross_mean_return is not None
    assert report.net_mean_return == pytest.approx(report.gross_mean_return - 0.0005)
    assert report.dsr is not None
    assert records[-1].metrics["dsr"] == report.dsr


def test_promotion_requires_all_three_strict_cutoffs() -> None:
    report = EvaluationReport(
        oos_ic_mean=0.1,
        ic_t_stat=2.1,
        hit_rate=None,
        gross_mean_return=None,
        net_mean_return=None,
        sharpe=None,
        dsr=0.951,
        n_trials=1,
        n_oos=30,
        n_trades=30,
        n_ic_periods=20,
        trial_id="trial",
    )
    assert passes_promotion_gate(report)
    assert not passes_promotion_gate(replace(report, n_ic_periods=19))
    assert not passes_promotion_gate(replace(report, oos_ic_mean=0))
    assert not passes_promotion_gate(replace(report, ic_t_stat=2))
    assert not passes_promotion_gate(replace(report, dsr=0.95))
    assert not passes_promotion_gate(replace(report, dsr=None))
