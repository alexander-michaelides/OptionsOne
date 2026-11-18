"""Run the live-compatible ridge model on purged out-of-sample events."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from json import dumps
from math import isfinite, sqrt
from statistics import fmean, stdev
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import Field, FiniteFloat

from optionsone.core.types import FeatureVector, Frozen
from optionsone.research.cv import purged_walk_forward_splits
from optionsone.research.labels import ForwardLabel, TripleBarrierLabel
from optionsone.research.metrics import (
    deflated_sharpe_ratio,
    hac_t_stat,
    hit_rate,
    period_rank_ic,
    sharpe_ratio,
)
from optionsone.research.registry import Parameter, TrialRecord, TrialRegistry
from optionsone.signals.model import DEFAULT_RIDGE_SPEC, LinearSignalModel, RidgeSpec

type EventKey = tuple[int, str, datetime]
type Outcome = ForwardLabel | TripleBarrierLabel

_EXCHANGE_TZ = ZoneInfo("America/New_York")


class EvaluationSpec(Frozen):
    ridge: RidgeSpec = DEFAULT_RIDGE_SPEC
    embargo: timedelta = Field(default=timedelta(minutes=1), ge=timedelta(0))
    cost_bps: FiniteFloat = Field(default=0.0, ge=0)
    hac_lags: int = Field(default=1, ge=0)
    market_adjusted: bool = False


DEFAULT_EVALUATION_SPEC = EvaluationSpec()


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    oos_ic_mean: float | None
    ic_t_stat: float | None
    hit_rate: float | None
    gross_mean_return: float | None
    net_mean_return: float | None
    sharpe: float | None
    dsr: float | None
    n_trials: int
    n_oos: int
    n_trades: int
    n_ic_periods: int
    trial_id: str


@dataclass(frozen=True, slots=True)
class _Sample:
    feature: FeatureVector
    label: Outcome
    target: float


def _align(
    feature_rows: Sequence[FeatureVector], labels: Mapping[EventKey, Outcome], spec: EvaluationSpec
) -> list[_Sample]:
    """Join by event identity so dropping unavailable labels cannot shift targets."""
    samples: list[_Sample] = []
    seen: set[EventKey] = set()
    for fv in sorted(feature_rows, key=lambda row: (row.as_of, row.ticker, row.article_id)):
        key = (fv.article_id, fv.ticker, fv.as_of)
        if key in seen:
            raise ValueError(f"duplicate feature event {key}")
        seen.add(key)
        label = labels.get(key)
        if label is None:
            continue
        if not fv.as_of < label.entry_time < label.end_time:
            raise ValueError(f"invalid label interval for event {key}")
        if isinstance(label, ForwardLabel) and label.end_time - label.entry_time != (
            spec.ridge.horizon
        ):
            raise ValueError(f"forward label horizon differs from model horizon for {key}")
        if spec.market_adjusted and isinstance(label, TripleBarrierLabel):
            raise ValueError("triple-barrier outcomes do not have adjusted returns")
        target = (
            label.market_adjusted_return
            if (spec.market_adjusted and isinstance(label, ForwardLabel))
            else label.raw_return
        )
        if target is not None:
            if not isfinite(target) or not isfinite(label.raw_return):
                raise ValueError(f"nonfinite label return for event {key}")
            samples.append(_Sample(fv, label, target))
    return samples


def _data_hash(samples: Sequence[_Sample]) -> str:
    """Hash the ordered, joined inputs including realized returns and intervals."""
    digest = sha256()
    for sample in samples:
        digest.update(
            dumps(
                (
                    sample.feature.article_id,
                    sample.feature.ticker,
                    sample.feature.as_of.isoformat(),
                    sample.feature.values,
                    sample.target,
                    sample.label.raw_return,
                    sample.label.entry_time.isoformat(),
                    sample.label.end_time.isoformat(),
                ),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate(
    feature_rows: Sequence[FeatureVector],
    labels: Mapping[EventKey, Outcome],
    n_splits: int,
    *,
    registry: TrialRegistry,
    spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> EvaluationReport:
    """Fit on purged training only and aggregate predictions by US exchange date.

    IC uses the selected target. Trade P&L always uses the raw stock return
    signed by the signal, with cost_bps charged once per round trip. Daily
    Sharpe uses equal-weight opportunities, including abstentions as zero.
    Use one registry for all candidate evaluations in the same strategy search.
    """
    samples = _align(feature_rows, labels, spec)
    trial_id = uuid4().hex
    params: dict[str, Parameter] = {
        "alpha": spec.ridge.alpha,
        "horizon_seconds": spec.ridge.horizon.total_seconds(),
        "threshold": spec.ridge.threshold,
        "feature_names": spec.ridge.feature_names,
        "n_splits": n_splits,
        "embargo_seconds": spec.embargo.total_seconds(),
        "cost_bps": spec.cost_bps,
        "hac_lags": spec.hac_lags,
        "market_adjusted": spec.market_adjusted,
    }
    trial_hash = _data_hash(samples)
    registry.append(
        TrialRecord(trial_id=trial_id, kind="trial_started", data_hash=trial_hash, params=params)
    )
    folds = purged_walk_forward_splits(
        [s.feature.as_of for s in samples],
        [s.label.end_time for s in samples],
        n_splits,
        embargo=spec.embargo,
    )
    scores: list[float] = []
    targets: list[float] = []
    periods: list[date] = []
    gross: list[float] = []
    daily_net: dict[date, list[float]] = {}
    cost = spec.cost_bps / 10_000
    for fold_number, fold in enumerate(folds):
        training = [samples[i] for i in fold.train]
        fold_hash = _data_hash(training)
        registry.append(
            TrialRecord(
                trial_id=trial_id,
                kind="fit_started",
                data_hash=fold_hash,
                params=params,
                fold=fold_number,
            )
        )
        model = LinearSignalModel.fit(
            [s.feature for s in training], [s.target for s in training], spec.ridge
        )
        errors: list[float] = []
        for index in fold.test:
            sample = samples[index]
            period = sample.feature.as_of.astimezone(_EXCHANGE_TZ).date()
            score = model.score(sample.feature)
            scores.append(score)
            targets.append(sample.target)
            periods.append(period)
            errors.append((score - sample.target) ** 2)
            signal = model.predict(sample.feature)
            net = 0.0
            if signal is not None:
                pnl = signal.side.sign * sample.label.raw_return
                gross.append(pnl)
                net = pnl - cost
            daily_net.setdefault(period, []).append(net)
        registry.append(
            TrialRecord(
                trial_id=trial_id,
                kind="fit_completed",
                data_hash=fold_hash,
                params=params,
                fold=fold_number,
                model_id=model.model_id,
                metrics={"oos_mse": fmean(errors)},
            )
        )

    ics = list(period_rank_ic(scores, targets, periods).values())
    returns = [fmean(day) for day in daily_net.values()]
    sharpe = sharpe_ratio(returns)
    history = registry.records()
    n_trials = sum(record.kind == "trial_started" for record in history)
    trial_sharpes = [
        value
        for record in history
        if record.kind == "trial_completed"
        if (value := record.metrics.get("sharpe")) is not None
    ]
    if sharpe is not None:
        trial_sharpes.append(sharpe)
    # Under a zero-Sharpe IID null use its sampling error as a dispersion floor.
    null_std = 1 / sqrt(len(returns) - 1) if len(returns) > 1 else 0.0
    dispersion = max(null_std, stdev(trial_sharpes) if len(trial_sharpes) > 1 else 0.0)
    gross_mean = fmean(gross) if gross else None
    report = EvaluationReport(
        oos_ic_mean=fmean(ics) if ics else None,
        ic_t_stat=hac_t_stat(ics, spec.hac_lags),
        hit_rate=hit_rate(gross),
        gross_mean_return=gross_mean,
        net_mean_return=gross_mean - cost if gross_mean is not None else None,
        sharpe=sharpe,
        dsr=deflated_sharpe_ratio(returns, n_trials, dispersion),
        n_trials=n_trials,
        n_oos=len(scores),
        n_trades=len(gross),
        n_ic_periods=len(ics),
        trial_id=trial_id,
    )
    registry.append(
        TrialRecord(
            trial_id=trial_id,
            kind="trial_completed",
            data_hash=trial_hash,
            params=params,
            metrics={
                "oos_ic_mean": report.oos_ic_mean,
                "ic_t_stat": report.ic_t_stat,
                "hit_rate": report.hit_rate,
                "gross_mean_return": report.gross_mean_return,
                "net_mean_return": report.net_mean_return,
                "sharpe": report.sharpe,
                "dsr": report.dsr,
            },
        )
    )
    return report


MIN_IC_PERIODS = 20


def passes_promotion_gate(report: EvaluationReport, min_ic_periods: int = MIN_IC_PERIODS) -> bool:
    """A HAC t-stat over a handful of daily periods is noise; require enough of them."""
    return (
        report.n_ic_periods >= min_ic_periods
        and report.oos_ic_mean is not None
        and report.oos_ic_mean > 0
        and report.ic_t_stat is not None
        and report.ic_t_stat > 2
        and report.dsr is not None
        and report.dsr > 0.95
    )
