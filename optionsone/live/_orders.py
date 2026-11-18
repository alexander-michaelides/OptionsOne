"""Track paper orders and reconcile filled inventory against broker snapshots."""

from collections import deque
from collections.abc import Sequence
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Final

from optionsone.core.clock import Clock
from optionsone.core.pit import PointInTimeBars
from optionsone.core.ports import Broker
from optionsone.core.types import OrderIntent, OrderSide, Position, Signal
from optionsone.execution.order_manager import OrderManager
from optionsone.live.journal import Journal, JournalRecord
from optionsone.risk.limits import RiskLimits, pre_trade_check
from optionsone.risk.sizing import SizingConfig, realized_bar_vol, size_shares

_MAX_MARK_AGE: Final = timedelta(minutes=2)


class PaperOrders:
    """Order manager plus the external inventory and journal used by the runner."""

    def __init__(
        self,
        broker: Broker,
        clock: Clock,
        journal: Journal,
        sizing: SizingConfig,
        limits: RiskLimits,
    ) -> None:
        self.manager = OrderManager(broker)
        self.submitted_at: deque[datetime] = deque()
        self.owned_quantities: dict[str, int] = {}
        self._order_ids: set[str] = set()
        self._broker = broker
        self._clock = clock
        self._journal = journal
        self._sizing = sizing
        self._limits = limits

    async def positions(self) -> Sequence[Position]:
        return await self._broker.positions()

    async def submit(self, order: OrderIntent, *, deadline: datetime | None = None) -> bool:
        try:
            sent = await self.manager.submit(order, deadline=deadline)
        except Exception as exc:
            self._journal.append(
                JournalRecord(
                    timestamp=self._clock.now(),
                    event="error",
                    reason="broker_submit_error",
                    symbol=order.symbol,
                    order=order,
                    error_type=type(exc).__name__,
                )
            )
            raise
        if not sent:
            return False
        self._order_ids.add(order.client_order_id)
        self.submitted_at.append(order.created_at)
        return True

    async def enter(
        self,
        signal: Signal,
        pit: PointInTimeBars,
        updated_at: datetime,
        equity: float,
    ) -> tuple[str, OrderIntent | None]:
        now = self._clock.now()
        deadline = signal.as_of + signal.horizon
        if signal.as_of > now:
            return "signal_not_yet_known", None
        if deadline <= now:
            return "expired_signal", None
        symbol = signal.ticker
        projected = self.manager.projected_quantities()
        if projected.get(symbol, 0) or self.manager.has_working_order(symbol):
            return "existing_position_or_order", None
        visible = pit.bars_asof(symbol)
        vol = realized_bar_vol(visible, self._sizing.vol_window)
        if vol is None or now - visible[-1].end > _MAX_MARK_AGE:
            return "missing_recent_bars", None
        qty = size_shares(signal, equity, visible[-1].close, vol, self._sizing)
        if not qty:
            return "size_below_one_share", None
        # The broker can deduplicate a replay across runner restarts.
        identity = (signal.article_id, updated_at.isoformat(), symbol)
        order_id = sha256(repr(identity).encode()).hexdigest()[:48]
        order = self.manager.entry_order(signal, qty, now).model_copy(
            update={"client_order_id": order_id},
        )
        prices = {
            ticker: mark.close
            for ticker in {*projected, symbol}
            if (mark := pit.last_bar(ticker)) is not None and now - mark.end <= _MAX_MARK_AGE
        }
        decision = pre_trade_check(
            order,
            self._limits,
            equity=equity,
            quantities=projected,
            prices=prices,
            submitted_at=self.submitted_at,
        )
        if not decision.allowed:
            return decision.reason, None
        if not await self.submit(order, deadline=deadline):
            return "duplicate_order", None
        return order.reason, order

    async def collect_fills(self) -> None:
        for fill in await self._broker.fills():
            if fill.client_order_id not in self._order_ids:
                continue
            self.manager.record_fill(fill)
            change = fill.qty if fill.side is OrderSide.BUY else -fill.qty
            quantity = self.owned_quantities.get(fill.symbol, 0) + change
            if quantity:
                self.owned_quantities[fill.symbol] = quantity
            else:
                self.owned_quantities.pop(fill.symbol, None)
            self._journal.append(
                JournalRecord(
                    timestamp=self._clock.now(),
                    event="fill",
                    reason="broker_fill",
                    symbol=fill.symbol,
                    fill=fill,
                )
            )

    async def cancel(self, reason: str) -> None:
        await self.manager.cancel_working()
        self._journal.append(
            JournalRecord(
                timestamp=self._clock.now(),
                event="cancel",
                reason=reason,
            )
        )

    async def exits(self, positions: Sequence[Position], *, flatten_reason: str | None) -> None:
        now = self._clock.now()
        due = self.owned_quantities.keys() if flatten_reason else self.manager.due_symbols(now)
        for position in positions:
            symbol = position.symbol
            if symbol not in due or not position.qty or self.manager.has_working_order(symbol):
                continue
            if position.qty != self.owned_quantities.get(symbol, 0):
                self._journal.append(
                    JournalRecord(
                        timestamp=now,
                        event="skip",
                        reason="foreign_position_in_symbol",
                        symbol=symbol,
                    )
                )
                continue
            order = self.manager.exit_order(symbol, now, flatten_reason or "horizon")
            if await self.submit(order):
                self._journal.append(
                    JournalRecord(
                        timestamp=self._clock.now(),
                        event="exit",
                        reason=order.reason,
                        symbol=symbol,
                        order=order,
                    )
                )
        while self.submitted_at and self.submitted_at[0] <= now - timedelta(minutes=1):
            self.submitted_at.popleft()
