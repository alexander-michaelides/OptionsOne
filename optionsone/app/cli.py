"""OptionsOne command line interface."""

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.offline import run_offline
from optionsone.classify.stub import DeterministicStubClassifier

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """News-driven intraday equity research and paper trading (no live trading)."""


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


@app.command()
def backtest(
    fixture: Annotated[bool, typer.Option(help="Use the seeded synthetic fixture.")] = False,
    days: Annotated[int, typer.Option(min=4)] = 10,
    seed: Annotated[int, typer.Option()] = 7,
    registry: Annotated[Path | None, typer.Option(help="Trial registry JSONL path.")] = None,
) -> None:
    """Classify news, evaluate out-of-sample IC on the first half, replay the second half."""
    if not fixture:
        typer.echo("Only --fixture is supported offline; see README for the store pipeline.")
        raise typer.Exit(2)
    data = synthetic_fixture(days=days, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = registry or Path(tmp) / "trials.jsonl"
        run = asyncio.run(
            run_offline(data.articles, data.bars, DeterministicStubClassifier(), path)
        )
    r = run.research_train
    bt = run.backtest.report
    typer.echo(f"classifications      {run.n_classifications}")
    typer.echo(f"model_id             {run.model_id}")
    typer.echo("-- research (purged walk-forward, train half) --")
    typer.echo(f"oos_rank_ic          {_fmt(r.oos_ic_mean, '+.4f')}")
    typer.echo(f"ic_hac_t             {_fmt(r.ic_t_stat, '+.2f')}")
    typer.echo(f"hit_rate             {_fmt(r.hit_rate, '.1%')}")
    typer.echo(f"dsr                  {_fmt(r.dsr, '.3f')}  (trials={r.n_trials})")
    typer.echo("-- backtest (out-of-sample half) --")
    typer.echo(f"trades               {len(bt.trades)}")
    typer.echo(f"gross_return         {bt.gross_return:+.4%}")
    typer.echo(f"net_return           {bt.net_return:+.4%}")
    typer.echo(f"sharpe_per_day       {bt.sharpe_per_day:+.2f}")
    typer.echo(f"max_drawdown         {bt.max_drawdown:.4%}")
    typer.echo(f"costs_total          {bt.costs.total:.2f}")
    typer.echo(f"rejected_orders      {len(run.backtest.rejected_orders)}")
