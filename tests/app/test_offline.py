from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

from typer.testing import CliRunner

from optionsone.app.cli import app
from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.offline import run_offline
from optionsone.classify.stub import DeterministicStubClassifier


async def test_planted_news_drift_is_recovered_out_of_sample(tmp_path: Path) -> None:
    data = synthetic_fixture(days=10, seed=7)
    run = await run_offline(
        data.articles, data.bars, DeterministicStubClassifier(), tmp_path / "t.jsonl"
    )
    assert run.research_train.oos_ic_mean is not None
    assert run.research_train.oos_ic_mean > 0.2
    assert run.backtest.report.trades
    assert run.backtest.report.net_return > 0


async def test_no_edge_without_planted_drift(tmp_path: Path) -> None:
    # Leakage detector: if features, labels or fills saw future bars, a null market would
    # still show positive IC on average. Single seeds are noisy, so test the distribution.
    ics: list[float] = []
    for seed in range(12):
        data = synthetic_fixture(days=10, seed=seed, drift_per_minute=0.0)
        run = await run_offline(
            data.articles, data.bars, DeterministicStubClassifier(), tmp_path / f"{seed}.jsonl"
        )
        assert run.research_train.oos_ic_mean is not None
        ics.append(run.research_train.oos_ic_mean)
        assert run.backtest.report.net_return <= run.backtest.report.gross_return
    assert abs(fmean(ics)) < 3 * stdev(ics) / sqrt(len(ics))


async def test_model_used_at_replay_start_ignores_bars_after_it(tmp_path: Path) -> None:
    # Gate review R1: 9 days puts the train/replay cut mid-session, so unpurged training
    # labels would end after the cut and the model would depend on post-cut prices.
    data = synthetic_fixture(days=9, seed=7)
    base = await run_offline(
        data.articles, data.bars, DeterministicStubClassifier(), tmp_path / "a.jsonl"
    )
    cut = base.replay_start
    bumped = tuple(
        b.model_copy(update={"close": b.close * 1.01, "high": b.high * 1.01})
        if b.start >= cut
        else b
        for b in data.bars
    )
    other = await run_offline(
        data.articles, bumped, DeterministicStubClassifier(), tmp_path / "b.jsonl"
    )
    assert other.model_id == base.model_id


def test_cli_backtest_fixture_prints_metrics() -> None:
    result = CliRunner().invoke(app, ["backtest", "--fixture"])
    assert result.exit_code == 0, result.output
    for key in ("trades", "net_return", "oos_rank_ic"):
        assert key in result.output
