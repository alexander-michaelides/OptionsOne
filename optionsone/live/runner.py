"""Run news decisions and account maintenance against a paper broker."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Final, Protocol
from zoneinfo import ZoneInfo

from optionsone.core.clock import Clock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.ports import Broker, Classifier, HistoricalBars, NewsSource
from optionsone.core.types import Article, OrderIntent, Signal
from optionsone.live._orders import PaperOrders
from optionsone.live.bars import LiveBarCache
from optionsone.live.journal import Journal, JournalRecord
from optionsone.risk.kill_switch import KillSwitch
from optionsone.risk.limits import RiskLimits
from optionsone.risk.sizing import SizingConfig
from optionsone.signals.decide import decide
from optionsone.signals.model import SignalModel

_EXCHANGE_TZ: Final = ZoneInfo("America/New_York")


class AccountEquity(Protocol):
    """Supply broker account equity including marked positions."""

    async def equity(self) -> float: ...


class PaperRunner:
    """Consume news and maintain one dedicated paper account until stopped."""

    def __init__(
        self,
        *,
        news_source: NewsSource,
        classifier: Classifier,
        bars: HistoricalBars,
        broker: Broker,
        equity_provider: AccountEquity,
        model: SignalModel,
        clock: Clock,
        sizing: SizingConfig,
        limits: RiskLimits,
        kill_switch: KillSwitch,
        journal: Journal,
        universe: Callable[[str], bool],
        is_regular_session: Callable[[datetime], bool],
        session_close: Callable[[date], datetime | None],
        tick_interval: float = 5.0,
        flatten_before: timedelta = timedelta(minutes=5),
        flatten_on_stop: bool = False,
        min_company_relevance: float = 0.0,
        min_novelty: float = 0.0,
        lookback_minutes: int = 45,
        wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isfinite(tick_interval) or tick_interval <= 0:
            raise ValueError("tick_interval must be positive and finite")
        if flatten_before < timedelta():
            raise ValueError("flatten_before must be nonnegative")
        self._news_source = news_source
        self._classifier = classifier
        self._bars = LiveBarCache(bars, clock, lookback_minutes=lookback_minutes)
        self._orders = PaperOrders(broker, clock, journal, sizing, limits)
        self._equity = equity_provider
        self._model = model
        self._clock = clock
        self._kill_switch = kill_switch
        self._journal = journal
        self._universe = universe
        self._is_regular_session = is_regular_session
        self._session_close = session_close
        self._tick_interval = tick_interval
        self._flatten_before = flatten_before
        self._flatten_on_stop = flatten_on_stop
        self._min_company_relevance = min_company_relevance
        self._min_novelty = min_novelty
        self._wait = wait
        self._stopped = asyncio.Event()
        self._account_lock = asyncio.Lock()
        self._seen: set[tuple[int, datetime]] = set()

    def stop(self) -> None:
        """Request graceful shutdown from the runner's asyncio loop."""
        self._stopped.set()

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Run both tasks until stopped, then cancel working orders and optionally flatten."""
        if stop_event is not None:
            if self._stopped.is_set():
                stop_event.set()
            self._stopped = stop_event
        stop = self._stopped
        for position in await self._orders.positions():
            if position.qty:
                self._journal.append(
                    JournalRecord(
                        timestamp=self._clock.now(),
                        event="skip",
                        reason="foreign_position_ignored",
                        symbol=position.symbol,
                    )
                )
        try:
            async with asyncio.TaskGroup() as group:
                news = group.create_task(self._consume(stop))
                ticker = group.create_task(self._tick(stop))
                await stop.wait()
                # A normal stop lets an in-flight broker operation finish before cancellation.
                async with self._account_lock:
                    news.cancel()
                    ticker.cancel()
        finally:
            await self._orders.cancel("shutdown")
            if self._flatten_on_stop:
                await self._orders.collect_fills()
                await self._orders.exits(
                    await self._orders.positions(),
                    flatten_reason="shutdown",
                )

    def _at_cutoff(self, now: datetime) -> bool:
        close = self._session_close(now.astimezone(_EXCHANGE_TZ).date())
        return close is not None and now >= close - self._flatten_before

    async def _tick(self, stop: asyncio.Event) -> None:
        last_flatten: tuple[date, str] | None = None
        while not stop.is_set():
            async with self._account_lock:
                await self._orders.collect_fills()
                equity = await self._equity.equity()
                now = self._clock.now()
                session = now.astimezone(_EXCHANGE_TZ).date()
                self._kill_switch.observe(session, equity)
                reason = (
                    "kill_switch"
                    if self._kill_switch.flatten_requested
                    else "session_close"
                    if self._at_cutoff(now)
                    else None
                )
                flatten = (session, reason) if reason else None
                if flatten is not None and flatten != last_flatten:
                    await self._orders.cancel(self._kill_switch.reason or flatten[1])
                    await self._orders.collect_fills()
                last_flatten = flatten
                positions = await self._orders.positions()
                if not stop.is_set():
                    await self._orders.exits(positions, flatten_reason=reason)
            await self._wait(self._tick_interval)

    async def _consume(self, stop: asyncio.Event) -> None:
        if stop.is_set():
            return
        stream = self._news_source.stream()
        try:
            async for article in stream:
                if stop.is_set():
                    break
                revision = (article.id, article.updated_at)
                base = JournalRecord(
                    timestamp=self._clock.now(),
                    event="skip",
                    reason="duplicate_article",
                    article_id=article.id,
                    updated_at=article.updated_at,
                )
                if revision in self._seen:
                    self._journal.append(base)
                    continue
                self._seen.add(revision)
                if article.knowledge_time > base.timestamp or not article.symbols:
                    reason = "article_not_yet_known" if article.symbols else "no_symbols"
                    self._journal.append(base.model_copy(update={"reason": reason}))
                    continue
                for symbol in dict.fromkeys(article.symbols):
                    if stop.is_set():
                        break
                    record = base.model_copy(update={"symbol": symbol})
                    await self._process(article, record, stop)
        finally:
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()

    async def _process(self, article: Article, record: JournalRecord, stop: asyncio.Event) -> None:
        symbol = record.symbol
        assert symbol is not None
        if not self._universe(symbol):
            self._journal.append(record.model_copy(update={"reason": "outside_universe"}))
            return
        try:
            classification = await self._classifier.classify(article, symbol)
        except Exception as exc:
            self._journal.append(
                record.model_copy(
                    update={
                        "timestamp": self._clock.now(),
                        "event": "error",
                        "reason": "classifier_error",
                        "error_type": type(exc).__name__,
                    }
                )
            )
            return
        symbols = {symbol, *self._orders.manager.projected_quantities()}
        pit = await self._bars.refresh(symbols)
        signal = decide(
            classification,
            pit,
            self._model,
            is_regular_session=self._is_regular_session,
            min_company_relevance=self._min_company_relevance,
            min_novelty=self._min_novelty,
        )
        if signal is None:
            reason = "session_cutoff" if self._at_cutoff(self._clock.now()) else "no_signal"
            order = None
        else:
            async with self._account_lock:
                reason, order = await self._entry(signal, pit, article.updated_at, stop)
        self._journal.append(
            record.model_copy(
                update={
                    "timestamp": self._clock.now(),
                    "event": "entry" if order is not None else "skip",
                    "reason": reason,
                    "signal": signal,
                    "order": order,
                }
            )
        )

    async def _entry(
        self,
        signal: Signal,
        pit: PointInTimeBars,
        updated_at: datetime,
        stop: asyncio.Event,
    ) -> tuple[str, OrderIntent | None]:
        equity = await self._equity.equity()
        positions = await self._orders.positions()
        now = self._clock.now()
        self._kill_switch.observe(now.astimezone(_EXCHANGE_TZ).date(), equity)
        if stop.is_set():
            return "shutdown", None
        if self._at_cutoff(now):
            return "session_cutoff", None
        if not self._is_regular_session(now):
            return "outside_regular_session", None
        if self._kill_switch.reason is not None:
            return self._kill_switch.reason, None
        symbol = signal.ticker
        if any(
            position.symbol == symbol
            and position.qty != self._orders.owned_quantities.get(symbol, 0)
            for position in positions
        ):
            return "foreign_position_in_symbol", None
        return await self._orders.enter(signal, pit, updated_at, equity)
