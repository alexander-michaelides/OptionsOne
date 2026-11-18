from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

from typer.testing import CliRunner

from optionsone.app.cli import app
from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.offline import run_offline
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.research.evaluate import passes_promotion_gate


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
        assert not passes_promotion_gate(run.research_train)
        assert run.backtest.report.net_return <= run.backtest.report.gross_return
    assert abs(fmean(ics)) < 3 * stdev(ics) / sqrt(len(ics))


def test_cli_backtest_fixture_prints_metrics() -> None:
    result = CliRunner().invoke(app, ["backtest", "--fixture"])
    assert result.exit_code == 0, result.output
    for key in ("trades", "net_return", "oos_rank_ic"):
        assert key in result.output
