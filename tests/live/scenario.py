"""Event-driven fake ports for exercising the full runner."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from pydantic import SecretStr

from optionsone.core.clock import SimClock as ManualClock
from optionsone.core.types import (
    Article,
    Bar,
    Classification,
    EffectType,
    EventType,
    Fill,
    OrderIntent,
    Position,
    TimeHorizon,
)
from optionsone.execution.order_manager import position_after_fill
from optionsone.ingest.calendar import is_regular_session, session_close
from optionsone.live.journal import Journal, JournalRecord
from optionsone.live.runner import PaperRunner
from optionsone.risk.kill_switch import KillSwitch, KillSwitchConfig
from optionsone.risk.limits import RiskLimits
from optionsone.risk.sizing import SizingConfig
from optionsone.signals.model import LinearSignalModel, RidgeSpec

START: Final = datetime(2026, 11, 18, 15, 30, tzinfo=UTC)
MINUTE: Final = timedelta(minutes=1)
HORIZON: Final = 2 * MINUTE
CUTOFF: Final = datetime(2026, 11, 18, 20, 55, tzinfo=UTC)
MODEL: Final = LinearSignalModel(
    spec=RidgeSpec(feature_names=("direction", "return_5m"), horizon=HORIZON),
    means=(0.0, 0.0),
    scales=(1.0, 1.0),
    weights=(1.0, 2.0),
    intercept=0.0,
    strength_scale=1.0,
)


class News:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[Article, asyncio.Event]] = asyncio.Queue()

    async def stream(self) -> AsyncGenerator[Article]:
        while True:
            article, handled = await self._queue.get()
            yield article
            handled.set()

    async def deliver(self, article: Article) -> None:
        handled = asyncio.Event()
        self._queue.put_nowait((article, handled))
        await handled.wait()


class PositiveClassifier:
    model_id = "classifier"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def classify(self, article: Article, ticker: str) -> Classification:
        self.calls.append((article.id, ticker))
        if article.headline == "failure":
            raise RuntimeError("test classifier outage")
        return Classification(
            article_id=article.id,
            ticker=ticker,
            model_id=self.model_id,
            prompt_version="test",
            event_type=EventType.EARNINGS,
            effect_type=EffectType.REVENUE,
            time_horizon=TimeHorizon.IMMEDIATE,
            p_positive=0.9,
            p_neutral=0.1,
            p_negative=0.0,
            materiality=1.0,
            novelty=1.0,
            certainty=1.0,
            company_relevance=1.0,
            market_relevance=0.0,
            already_known=0.0,
            knowledge_time=article.knowledge_time,
        )


class Bars:
    def __init__(self) -> None:
        self.last_bars: tuple[Bar, ...] = ()
        self.requests: list[tuple[tuple[str, ...], datetime, datetime]] = []

    async def fetch(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> Sequence[Bar]:
        self.requests.append((tuple(symbols), start, end))
        last_start = end.replace(second=0, microsecond=0)
        self.last_bars = tuple(
            Bar(
                symbol=symbol,
                start=last_start - offset * MINUTE,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=100,
            )
            for symbol in symbols
            for offset in range(-1, int((last_start - start) // MINUTE) + 1)
            for price in (150.0 - offset / 100 if offset >= 0 else 500.0,)
        )
        return self.last_bars


class PaperBroker:
    def __init__(self, clock: ManualClock) -> None:
        self.orders: list[OrderIntent] = []
        self.working: dict[str, OrderIntent] = {}
        self.inventory: dict[str, Position] = {}
        self.pending_fills: list[Fill] = []
        self.account_value = 100_000.0
        self._clock = clock

    async def submit(self, order: OrderIntent) -> None:
        self.orders.append(order)
        self.working[order.client_order_id] = order

    async def fills(self) -> Sequence[Fill]:
        fills = tuple(self.pending_fills)
        self.pending_fills.clear()
        return fills

    async def positions(self) -> Sequence[Position]:
        return tuple(self.inventory.values())

    async def equity(self) -> float:
        return self.account_value

    async def cancel(self, client_order_ids: Sequence[str]) -> None:
        for client_order_id in client_order_ids:
            self.working.pop(client_order_id, None)

    def fill(self, order: OrderIntent) -> None:
        self.working.pop(order.client_order_id)
        fill = Fill(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=150.0,
            fee=0,
            filled_at=self._clock.now(),
        )
        self.pending_fills.append(fill)
        position = position_after_fill(self.inventory.get(order.symbol), fill)
        if position is None:
            self.inventory.pop(order.symbol)
        else:
            self.inventory[order.symbol] = position


class Scenario:
    def __init__(self, path: Path, *, flatten_on_stop: bool = False) -> None:
        self.clock = ManualClock(START)
        self.news = News()
        self.classifier = PositiveClassifier()
        self.bars = Bars()
        self.broker = PaperBroker(self.clock)
        self.kill_switch = KillSwitch(KillSwitchConfig(reset_code=SecretStr("reset")))
        self.path = path
        self.stop_event = asyncio.Event()
        self._ticked = asyncio.Event()
        self._advance_tick = asyncio.Event()
        self.runner = PaperRunner(
            news_source=self.news,
            classifier=self.classifier,
            bars=self.bars,
            broker=self.broker,
            equity_provider=self.broker,
            model=MODEL,
            clock=self.clock,
            sizing=SizingConfig(),
            limits=RiskLimits(max_open_positions=1),
            kill_switch=self.kill_switch,
            journal=Journal(self.path),
            universe={"ACME", "OTHER"}.__contains__,
            is_regular_session=is_regular_session,
            session_close=session_close,
            flatten_on_stop=flatten_on_stop,
            wait=self._wait,
        )

    async def _wait(self, _: float) -> None:
        self._ticked.set()
        await self._advance_tick.wait()
        self._advance_tick.clear()

    @asynccontextmanager
    async def running(self) -> AsyncGenerator[None]:
        async with asyncio.timeout(2), asyncio.TaskGroup() as group:
            group.create_task(self.runner.run(self.stop_event))
            try:
                await self._ticked.wait()
                yield
            finally:
                self.runner.stop()

    async def tick(self, when: datetime) -> None:
        self._ticked.clear()
        self.clock.advance_to(when)
        self._advance_tick.set()
        await self._ticked.wait()

    def article(
        self, article_id: int = 1, symbol: str = "ACME", headline: str = "growth"
    ) -> Article:
        now = self.clock.now()
        return Article(
            id=article_id,
            headline=headline,
            symbols=(symbol,),
            published_at=now,
            updated_at=now,
            knowledge_time=now,
        )

    def records(self) -> list[JournalRecord]:
        return [
            JournalRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]
