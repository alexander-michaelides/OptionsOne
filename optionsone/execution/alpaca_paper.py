"""Alpaca paper REST adapter for a dedicated whole-share equity account."""

from collections.abc import Sequence
from typing import Final
from urllib.parse import quote

import httpx
from anyio import Lock
from pydantic import AliasChoices, AwareDatetime, BaseModel, Field, SecretStr, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

from optionsone.core.clock import Clock, LiveClock
from optionsone.core.ports import Broker
from optionsone.core.types import Fill, OrderIntent, OrderSide, Position, UtcDatetime

PAPER_BASE_URL: Final = "https://paper-api.alpaca.markets"
_ORDERS: Final = "/v2/orders"
_POSITIONS: Final = "/v2/positions"


class LiveTradingForbidden(ValueError):
    def __init__(self, url: str) -> None:
        super().__init__(f"only the Alpaca paper trading URL is allowed, got {url!r}")


class LiveKeyForbidden(LiveTradingForbidden):
    """Alpaca live-account keys start with 'AK'; paper keys start with 'PK'."""

    def __init__(self) -> None:
        ValueError.__init__(
            self, "refusing an Alpaca live-account key (AK...); use a paper key (PK...)"
        )


class MissingPaperCredentials(ValueError):
    """The paper account key and secret must both be configured."""


class PaperCancellationError(RuntimeError):
    def __init__(self, order_ids: tuple[str, ...]) -> None:
        self.order_ids = order_ids
        super().__init__(f"paper orders could not be canceled: {order_ids}")


class AlpacaPaperSettings(BaseSettings):
    """Paper credentials from the environment or ./.env.

    ALPACA_PAPER_* wins; ALPACA_API_KEY / ALPACA_SECRET_KEY are accepted when they are paper keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, populate_by_name=True
    )

    base_url: str = Field(
        default=PAPER_BASE_URL, validation_alias=AliasChoices("ALPACA_PAPER_BASE_URL", "base_url")
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("ALPACA_PAPER_API_KEY", "ALPACA_API_KEY", "api_key"),
    )
    secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("ALPACA_PAPER_SECRET_KEY", "ALPACA_SECRET_KEY", "secret_key"),
    )


class AccountSummary(BaseModel):
    status: str
    currency: str
    equity: float
    buying_power: float
    trading_blocked: bool


class MarketClock(BaseModel):
    """Alpaca reports clock times with an America/New_York offset."""

    is_open: bool
    timestamp: AwareDatetime
    next_open: AwareDatetime
    next_close: AwareDatetime


class OrderStatus(BaseModel):
    id: str
    client_order_id: str
    symbol: str
    status: str
    type: str
    limit_price: float | None = None


class _RemoteOrder(BaseModel):
    id: str
    client_order_id: str


class _RemotePosition(BaseModel):
    symbol: str
    qty: int
    avg_entry_price: float


class _Activity(BaseModel):
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    transaction_time: UtcDatetime


class AlpacaPaperBroker(Broker):
    """Reconcile before submitting and return incremental account fill activities."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        settings: AlpacaPaperSettings | None = None,
        clock: Clock | None = None,
    ) -> None:
        if base_url is not None and base_url != PAPER_BASE_URL:
            raise LiveTradingForbidden(base_url)
        self._settings = settings if settings is not None else AlpacaPaperSettings()
        if self._settings.base_url != PAPER_BASE_URL:
            raise LiveTradingForbidden(self._settings.base_url)
        if self._settings.api_key.get_secret_value().upper().startswith("AK"):
            raise LiveKeyForbidden()
        self._clock = clock if clock is not None else LiveClock()
        self._lock = Lock()
        self._started_after: str | None = None
        self._activity_cursor: str | None = None
        self._known: dict[str, str] = {}

    def _client(self) -> httpx.AsyncClient:
        key = self._settings.api_key.get_secret_value()
        secret = self._settings.secret_key.get_secret_value()
        if not key or not secret:
            raise MissingPaperCredentials("set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY")
        return httpx.AsyncClient(
            base_url=PAPER_BASE_URL,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(15, connect=5),
        )

    async def _positions(self, client: httpx.AsyncClient) -> tuple[Position, ...]:
        response = (await client.get(_POSITIONS)).raise_for_status()
        positions = TypeAdapter(list[_RemotePosition]).validate_json(response.content)
        return tuple(
            Position(symbol=item.symbol, qty=item.qty, avg_price=item.avg_entry_price)
            for item in positions
        )

    async def _reconcile(self, client: httpx.AsyncClient) -> tuple[Position, ...]:
        start = self._clock.now().isoformat().replace("+00:00", "Z")
        params = {"status": "open", "limit": "500", "direction": "desc"}
        while True:
            response = (await client.get(_ORDERS, params=params)).raise_for_status()
            orders = TypeAdapter(list[_RemoteOrder]).validate_json(response.content)
            self._known.update((order.client_order_id, order.id) for order in orders)
            if len(orders) < 500:
                break
            params["before_order_id"] = orders[-1].id
        positions = await self._positions(client)
        if self._started_after is None:
            self._started_after = start
        return positions

    async def reconcile(self) -> tuple[Position, ...]:
        async with self._lock, self._client() as client:
            return await self._reconcile(client)

    async def submit(self, order: OrderIntent) -> None:
        async with self._lock, self._client() as client:
            if self._started_after is None:
                await self._reconcile(client)
            if order.client_order_id in self._known:
                return
            response = await client.post(
                _ORDERS,
                json={
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": str(order.qty),
                    "type": "market",
                    "time_in_force": "day",
                },
            )
            if response.status_code == 422:
                existing = await client.get(
                    "/v2/orders:by_client_order_id",
                    params={"client_order_id": order.client_order_id},
                )
                if existing.is_success:
                    response = existing
            remote = _RemoteOrder.model_validate_json(response.raise_for_status().content)
            self._known[remote.client_order_id] = remote.id

    async def fills(self) -> Sequence[Fill]:
        async with self._lock, self._client() as client:
            if self._started_after is None:
                await self._reconcile(client)
            params = {
                "activity_types": "FILL",
                "direction": "asc",
                "page_size": "100",
                "after": self._started_after,
            }
            cursor = self._activity_cursor
            client_ids = {remote: local for local, remote in self._known.items()}
            fills: list[Fill] = []
            while True:
                if cursor is not None:
                    params["page_token"] = cursor
                response = (
                    await client.get("/v2/account/activities", params=params)
                ).raise_for_status()
                activities = TypeAdapter(list[_Activity]).validate_json(response.content)
                for activity in activities:
                    client_id = client_ids.get(activity.order_id)
                    if client_id is None:
                        response = (
                            await client.get(f"{_ORDERS}/{quote(activity.order_id, safe='')}")
                        ).raise_for_status()
                        order = _RemoteOrder.model_validate_json(response.content)
                        client_id = client_ids[order.id] = order.client_order_id
                        self._known[client_id] = order.id
                    fills.append(
                        Fill(
                            client_order_id=client_id,
                            symbol=activity.symbol,
                            side=activity.side,
                            qty=activity.qty,
                            price=activity.price,
                            fee=0.0,
                            filled_at=activity.transaction_time,
                        )
                    )
                    cursor = activity.id
                if len(activities) < 100:
                    self._activity_cursor = cursor
                    return fills

    async def positions(self) -> Sequence[Position]:
        async with self._lock, self._client() as client:
            if self._started_after is None:
                return await self._reconcile(client)
            return await self._positions(client)

    async def account(self) -> AccountSummary:
        async with self._client() as client:
            response = (await client.get("/v2/account")).raise_for_status()
            return AccountSummary.model_validate_json(response.content)

    async def market_clock(self) -> MarketClock:
        async with self._client() as client:
            response = (await client.get("/v2/clock")).raise_for_status()
            return MarketClock.model_validate_json(response.content)

    async def submit_limit(
        self, symbol: str, side: OrderSide, qty: int, limit_price: float, client_order_id: str
    ) -> OrderStatus:
        """Day limit order; used by the paper smoke test, not by the strategy."""
        async with self._lock, self._client() as client:
            response = await client.post(
                _ORDERS,
                json={
                    "client_order_id": client_order_id,
                    "symbol": symbol,
                    "side": side.value,
                    "qty": str(qty),
                    "type": "limit",
                    "limit_price": f"{limit_price:.2f}",
                    "time_in_force": "day",
                },
            )
            order = OrderStatus.model_validate_json(response.raise_for_status().content)
            self._known[order.client_order_id] = order.id
            return order

    async def order(self, order_id: str) -> OrderStatus:
        async with self._client() as client:
            path = f"{_ORDERS}/{quote(order_id, safe='')}"
            response = (await client.get(path)).raise_for_status()
            return OrderStatus.model_validate_json(response.content)

    async def cancel_order(self, order_id: str) -> None:
        async with self._lock, self._client() as client:
            (await client.delete(f"{_ORDERS}/{quote(order_id, safe='')}")).raise_for_status()

    async def cancel(self, client_order_ids: Sequence[str]) -> None:
        """Cancel only orders this process submitted; never DELETE /v2/orders (account-wide)."""
        failures: list[str] = []
        async with self._lock, self._client() as client:
            for client_order_id in client_order_ids:
                remote = self._known.get(client_order_id)
                if remote is None:
                    found = await client.get(
                        "/v2/orders:by_client_order_id",
                        params={"client_order_id": client_order_id},
                    )
                    if found.status_code == 404:
                        continue
                    remote = _RemoteOrder.model_validate_json(found.raise_for_status().content).id
                response = await client.delete(f"{_ORDERS}/{quote(remote, safe='')}")
                # 422: no longer cancelable (already filled or canceled) is a settled state.
                if not (response.is_success or response.status_code in (404, 422)):
                    failures.append(client_order_id)
        if failures:
            raise PaperCancellationError(tuple(failures))
