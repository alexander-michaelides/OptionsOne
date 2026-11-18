"""OptionsOne command line interface."""

import asyncio
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from optionsone.app.classifiers import make_classifier
from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.offline import OfflineRun, run_offline
from optionsone.app.paper_cmd import (
    UnpromotedModel,
    load_model,
    paper_check,
    paper_smoke,
    run_paper,
)
from optionsone.app.research_cmd import (
    DayProgress,
    EmptyStoreError,
    backfill,
    backtest_store,
    research,
)
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.clock import LiveClock
from optionsone.execution.alpaca_paper import AlpacaPaperBroker
from optionsone.ingest.alpaca_bars import AlpacaHistoricalBars
from optionsone.ingest.alpaca_news import AlpacaHistoricalNews
from optionsone.risk.limits import RiskLimits
from optionsone.risk.sizing import SizingConfig

# Never render local variables in tracebacks: they can hold credentials.
app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """News-driven intraday equity research and paper trading (no live trading)."""


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


DATA = Path("data")


class ClassifierKind(StrEnum):
    STUB = "stub"
    LAYA = "laya"
    HOSTED = "hosted"


def _utc(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def _echo_day(progress: DayProgress) -> None:
    if progress.resumed:
        typer.echo(f"{progress.day}  already done, skipped")
    else:
        typer.echo(
            f"{progress.day}  articles +{progress.articles_added:<6} bars +{progress.bars_added}"
        )


@app.command("backfill")
def backfill_cmd(
    start: Annotated[str, typer.Option(help="UTC start date, e.g. 2026-08-01")],
    end: Annotated[str, typer.Option(help="UTC end date (exclusive)")],
    symbols: Annotated[str, typer.Option(help="Comma-separated universe; empty = all news")] = "",
    store: Annotated[Path, typer.Option()] = DATA / "store",
) -> None:
    """Download Alpaca news and IEX 1-minute bars into the Parquet store."""
    universe = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    bar_source = AlpacaHistoricalBars()
    report = asyncio.run(
        backfill(
            store,
            universe,
            _utc(start),
            _utc(end),
            news=AlpacaHistoricalNews(),
            bars=bar_source,
            on_day=_echo_day,
        )
    )
    typer.echo(f"articles fetched/added  {report.articles_fetched}/{report.articles_added}")
    typer.echo(f"bars fetched/added      {report.bars_fetched}/{report.bars_added}")
    skipped = sorted(bar_source.unsupported_symbols)
    if skipped:
        shown = ", ".join(skipped[:12]) + (" ..." if len(skipped) > 12 else "")
        typer.echo(f"symbols skipped         {len(skipped)} non-US or rejected ({shown})")
    v = report.validation
    typer.echo(
        f"bar quality             gaps={len(v.gaps)} nonpositive={len(v.nonpositive_prices)} "
        f"duplicates={len(v.duplicates)} stale_runs={len(v.stale_runs)}"
    )


@app.command("research")
def research_cmd(
    start: Annotated[str, typer.Option()],
    end: Annotated[str, typer.Option()],
    classifier: Annotated[ClassifierKind, typer.Option()] = ClassifierKind.STUB,
    symbols: Annotated[str, typer.Option(help="Restrict to these tickers")] = "",
    store: Annotated[Path, typer.Option()] = DATA / "store",
    model_out: Annotated[Path, typer.Option()] = DATA / "models" / "model.json",
) -> None:
    """Classify stored news, evaluate with purged walk-forward CV, fit and save the model."""
    model_out.parent.mkdir(parents=True, exist_ok=True)
    universe = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        result = asyncio.run(
            research(
                store,
                _utc(start),
                _utc(end),
                make_classifier(classifier.value, DATA / "cache" / "classifications.sqlite"),
                model_out=model_out,
                report_out=model_out.with_suffix(".report.json"),
                registry_path=DATA / "trials.jsonl",
                symbols=universe,
            )
        )
    except EmptyStoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    e = result.evaluation
    typer.echo(f"model_id             {result.model_id}  -> {model_out}")
    typer.echo(f"oos_rank_ic          {_fmt(e.oos_ic_mean, '+.4f')}  (periods={e.n_ic_periods})")
    typer.echo(f"ic_hac_t             {_fmt(e.ic_t_stat, '+.2f')}")
    typer.echo(f"net_mean_return      {_fmt(e.net_mean_return, '+.5f')}")
    typer.echo(f"dsr                  {_fmt(e.dsr, '.3f')}  (trials={e.n_trials})")
    typer.echo(f"promotion_gate       {'PASS' if result.passes_promotion_gate else 'FAIL'}")


@app.command()
def backtest(
    fixture: Annotated[bool, typer.Option(help="Use the seeded synthetic fixture.")] = False,
    start: Annotated[str | None, typer.Option(help="Store mode: UTC start date")] = None,
    end: Annotated[str | None, typer.Option(help="Store mode: UTC end date (exclusive)")] = None,
    symbols: Annotated[str, typer.Option(help="Store mode: restrict to these tickers")] = "",
    classifier: Annotated[ClassifierKind, typer.Option()] = ClassifierKind.STUB,
    store: Annotated[Path, typer.Option()] = DATA / "store",
    train_fraction: Annotated[float, typer.Option(min=0.1, max=0.9)] = 0.5,
    initial_cash: Annotated[float, typer.Option(min=100.0)] = 100_000.0,
    days: Annotated[int, typer.Option(min=4, help="Fixture mode: trading days")] = 10,
    seed: Annotated[int, typer.Option(help="Fixture mode: RNG seed")] = 7,
) -> None:
    """Fit on the first part of the timeline, then replay the rest event by event.

    --fixture uses a synthetic market; --start/--end replay your backfilled store.
    """
    if fixture == (start is not None or end is not None):
        typer.echo("error: pass either --fixture or both --start and --end", err=True)
        raise typer.Exit(2)
    if fixture:
        data = synthetic_fixture(days=days, seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            run = asyncio.run(
                run_offline(
                    data.articles,
                    data.bars,
                    DeterministicStubClassifier(),
                    Path(tmp) / "trials.jsonl",
                    train_fraction=train_fraction,
                    initial_cash=initial_cash,
                )
            )
    else:
        if start is None or end is None:
            typer.echo("error: store mode needs both --start and --end", err=True)
            raise typer.Exit(2)
        universe = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        try:
            run = asyncio.run(
                backtest_store(
                    store,
                    _utc(start),
                    _utc(end),
                    make_classifier(classifier.value, DATA / "cache" / "classifications.sqlite"),
                    registry_path=DATA / "trials.jsonl",
                    symbols=universe,
                    train_fraction=train_fraction,
                    initial_cash=initial_cash,
                )
            )
        except EmptyStoreError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
    _print_run(run)


def _print_run(run: OfflineRun) -> None:
    r = run.research_train
    bt = run.backtest.report
    typer.echo(f"classifications      {run.n_classifications}")
    typer.echo(f"model_id             {run.model_id}")
    typer.echo(f"replay_start         {run.replay_start:%Y-%m-%d %H:%M}Z")
    typer.echo("-- research (purged walk-forward, training part) --")
    typer.echo(f"oos_rank_ic          {_fmt(r.oos_ic_mean, '+.4f')}  (periods={r.n_ic_periods})")
    typer.echo(f"ic_hac_t             {_fmt(r.ic_t_stat, '+.2f')}")
    typer.echo(f"hit_rate             {_fmt(r.hit_rate, '.1%')}")
    typer.echo(f"dsr                  {_fmt(r.dsr, '.3f')}  (trials={r.n_trials})")
    typer.echo("-- backtest (out-of-sample replay) --")
    typer.echo(f"signals              {run.backtest.signals}")
    for reason, count in run.backtest.skipped:
        typer.echo(f"  skipped            {count} {reason}")
    typer.echo(f"trades               {len(bt.trades)}")
    typer.echo(f"gross_return         {bt.gross_return:+.4%}")
    typer.echo(f"net_return           {bt.net_return:+.4%}")
    typer.echo(
        "net_at_cost_x0/1/2   " + " / ".join(f"{value:+.4%}" for _, value in bt.cost_sensitivity)
    )
    typer.echo(f"sharpe_per_day       {bt.sharpe_per_day:+.2f}")
    typer.echo(f"max_drawdown         {bt.max_drawdown:.4%}")
    typer.echo(f"turnover             {bt.turnover:.2f}x")
    typer.echo(f"costs_total          {bt.costs.total:.2f}")
    typer.echo(f"rejected_orders      {len(run.backtest.rejected_orders)}")
    typer.echo(f"open_at_end          {len(run.backtest.open_positions)}")


def _bucket(value: float) -> str:
    """Summarize money without printing exact balances."""
    return f"~{round(value, -3):,.0f}"


@app.command("paper-check")
def paper_check_cmd() -> None:
    """Verify paper credentials, market clock, positions and IEX data access."""
    result = asyncio.run(paper_check(AlpacaPaperBroker(), AlpacaHistoricalBars(), LiveClock()))
    a, m = result.account, result.clock
    typer.echo("endpoint             paper-api.alpaca.markets (paper only)")
    typer.echo(f"account_status       {a.status}")
    typer.echo(f"trading_blocked      {a.trading_blocked}")
    typer.echo(f"equity               {_bucket(a.equity)} {a.currency}")
    typer.echo(f"buying_power         {_bucket(a.buying_power)} {a.currency}")
    typer.echo(f"market_open          {m.is_open}  next_open {m.next_open:%Y-%m-%d %H:%M %z}")
    typer.echo(f"open_positions       {result.open_positions}")
    typer.echo(f"spy_iex_bars_4d      {result.recent_bars}")
    if a.status != "ACTIVE" or a.trading_blocked or result.recent_bars == 0:
        raise typer.Exit(1)


@app.command("paper-smoke")
def paper_smoke_cmd(symbol: Annotated[str, typer.Option()] = "SPY") -> None:
    """Place a non-marketable 1-share paper limit order, then cancel it."""
    result = asyncio.run(
        paper_smoke(AlpacaPaperBroker(), AlpacaHistoricalBars(), LiveClock(), symbol=symbol)
    )
    typer.echo(f"symbol               {result.symbol}")
    typer.echo(f"reference_price      {result.reference_price:.2f}")
    typer.echo(f"limit_price          {result.limit_price:.2f}")
    typer.echo(f"status_transitions   {' -> '.join(result.statuses)}")
    typer.echo(f"result               {'PASS' if result.ok else 'FAIL'}")
    if not result.ok:
        raise typer.Exit(1)


@app.command("paper")
def paper_cmd(
    symbols: Annotated[str, typer.Option(help="Comma-separated tradable universe")],
    model: Annotated[Path, typer.Option()] = DATA / "models" / "model.json",
    classifier: Annotated[ClassifierKind, typer.Option()] = ClassifierKind.STUB,
    allow_unpromoted: Annotated[
        bool, typer.Option(help="Trade a model that failed the gate")
    ] = False,
    max_minutes: Annotated[float | None, typer.Option(help="Stop after N minutes")] = None,
    flatten_on_stop: Annotated[bool, typer.Option()] = False,
    max_position_pct: Annotated[float, typer.Option(min=0.001, max=1.0)] = 0.10,
    max_open_positions: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Run the news -> classify -> decide -> risk -> PAPER order loop (Ctrl-C to stop)."""
    universe = frozenset(s.strip().upper() for s in symbols.split(",") if s.strip())
    if not universe:
        raise typer.BadParameter("--symbols must name at least one ticker")
    journal = DATA / "journal" / f"paper-{LiveClock().now():%Y%m%dT%H%M%SZ}.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    try:
        signal_model = load_model(model, allow_unpromoted=allow_unpromoted)
    except (UnpromotedModel, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"paper runner: model {signal_model.model_id[:12]} universe {sorted(universe)}")
    typer.echo(f"journal: {journal}")
    asyncio.run(
        run_paper(
            model=signal_model,
            classifier=make_classifier(classifier.value, DATA / "cache" / "classifications.sqlite"),
            universe=universe,
            journal_path=journal,
            max_minutes=max_minutes,
            flatten_on_stop=flatten_on_stop,
            limits=RiskLimits(
                max_position_notional_pct=max_position_pct, max_open_positions=max_open_positions
            ),
            sizing=SizingConfig(max_position_pct=max_position_pct),
        )
    )
    typer.echo("stopped. Working orders cancelled; see the journal for every decision.")
