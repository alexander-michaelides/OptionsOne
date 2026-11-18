"""Paper-account commands: diagnostics, smoke test, and the live paper runner wiring."""

import asyncio
import json
import secrets
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import anyio
from pydantic import SecretStr

from optionsone.core.clock import Clock, LiveClock
from optionsone.core.ports import Classifier, HistoricalBars
from optionsone.core.types import OrderSide
from optionsone.execution.alpaca_paper import AccountSummary, AlpacaPaperBroker, MarketClock
from optionsone.ingest.alpaca_bars import AlpacaHistoricalBars
from optionsone.ingest.alpaca_news import AlpacaNewsStream
from optionsone.ingest.calendar import is_regular_session, session_close
from optionsone.live.journal import Journal
from optionsone.live.runner import PaperRunner
from optionsone.risk.kill_switch import KillSwitch, KillSwitchConfig
from optionsone.risk.limits import RiskLimits
from optionsone.risk.sizing import SizingConfig
from optionsone.signals.model import LinearSignalModel

_OPEN_STATES = frozenset({"new", "accepted", "pending_new"})


@dataclass(frozen=True, slots=True)
class PaperCheck:
    account: AccountSummary
    clock: MarketClock
    open_positions: int
    recent_bars: int


@dataclass(frozen=True, slots=True)
class SmokeResult:
    symbol: str
    reference_price: float
    limit_price: float
    statuses: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            bool(self.statuses)
            and self.statuses[0] in _OPEN_STATES
            and self.statuses[-1] == "canceled"
        )


async def paper_check(broker: AlpacaPaperBroker, bars: HistoricalBars, clock: Clock) -> PaperCheck:
    account = await broker.account()
    market = await broker.market_clock()
    positions = await broker.reconcile()
    now = clock.now()
    recent = await bars.fetch(["SPY"], now - timedelta(days=4), now - timedelta(minutes=16))
    return PaperCheck(account, market, len(positions), len(recent))


async def paper_smoke(
    broker: AlpacaPaperBroker,
    bars: HistoricalBars,
    clock: Clock,
    *,
    symbol: str = "SPY",
    attempts: int = 20,
    wait: Callable[[float], Awaitable[None]] | None = None,
) -> SmokeResult:
    """Place a 1-share buy limit at half the last price (cannot fill), then cancel it."""
    pause = wait if wait is not None else anyio.sleep
    now = clock.now()
    recent = await bars.fetch([symbol], now - timedelta(days=4), now - timedelta(minutes=16))
    if not recent:
        raise RuntimeError(f"no recent IEX bars for {symbol}; cannot pick a safe limit price")
    reference = recent[-1].close
    limit = round(reference * 0.5, 2)
    order = await broker.submit_limit(
        symbol, OrderSide.BUY, 1, limit, f"o1-smoke-{uuid4().hex[:16]}"
    )
    statuses = [order.status]
    await broker.cancel_order(order.id)
    for _ in range(attempts):
        current = await broker.order(order.id)
        if current.status != statuses[-1]:
            statuses.append(current.status)
        if current.status in {"canceled", "filled", "rejected", "expired"}:
            break
        await pause(0.5)  # remote state transition; bounded by attempts
    return SmokeResult(symbol, reference, limit, tuple(statuses))


class UnpromotedModel(RuntimeError):
    """The model's research report did not pass the promotion gate."""


class BrokerEquity:
    def __init__(self, broker: AlpacaPaperBroker) -> None:
        self._broker = broker

    async def equity(self) -> float:
        return (await self._broker.account()).equity


def load_model(path: Path, *, allow_unpromoted: bool) -> LinearSignalModel:
    model = LinearSignalModel.from_json(path.read_text())
    report_path = path.with_suffix(".report.json")
    promoted = report_path.exists() and bool(
        json.loads(report_path.read_text()).get("passes_promotion_gate")
    )
    if not promoted and not allow_unpromoted:
        raise UnpromotedModel(
            f"{path} has not passed the promotion gate ({report_path}); "
            "pass --allow-unpromoted to paper-trade it anyway"
        )
    return model


async def run_paper(
    *,
    model: LinearSignalModel,
    classifier: Classifier,
    universe: frozenset[str],
    journal_path: Path,
    max_minutes: float | None,
    flatten_on_stop: bool,
    limits: RiskLimits,
    sizing: SizingConfig,
) -> str:
    """Run until SIGINT/SIGTERM or max_minutes. Returns the kill-switch reset code used."""
    clock = LiveClock()
    broker = AlpacaPaperBroker(clock=clock)
    reset_code = secrets.token_hex(8)
    runner = PaperRunner(
        news_source=AlpacaNewsStream(clock=clock),
        classifier=classifier,
        bars=AlpacaHistoricalBars(),
        broker=broker,
        equity_provider=BrokerEquity(broker),
        model=model,
        clock=clock,
        sizing=sizing,
        limits=limits,
        kill_switch=KillSwitch(KillSwitchConfig(reset_code=SecretStr(reset_code))),
        journal=Journal(journal_path),
        universe=lambda symbol: symbol in universe,
        is_regular_session=is_regular_session,
        session_close=session_close,
        flatten_on_stop=flatten_on_stop,
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.stop)
    if max_minutes is not None:
        loop.call_later(max_minutes * 60, runner.stop)
    try:
        await runner.run()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    return reset_code
