from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

from optionsone.app.paper_cmd import paper_smoke
from optionsone.core.clock import SimClock
from optionsone.core.types import Bar, OrderSide
from optionsone.execution.alpaca_paper import AlpacaPaperBroker, OrderStatus

NOW = datetime(2026, 11, 18, 20, tzinfo=UTC)


class _Bars:
    async def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[Bar]:
        return [
            Bar(
                symbol="SPY",
                start=end - timedelta(minutes=1),
                open=700,
                high=701,
                low=699,
                close=700,
                volume=10,
            )
        ]


class _Broker:
    def __init__(self, final: str) -> None:
        self.final = final
        self.submitted: list[tuple[str, OrderSide, int, float]] = []
        self.cancelled: list[str] = []

    async def submit_limit(
        self, symbol: str, side: OrderSide, qty: int, limit_price: float, client_order_id: str
    ) -> OrderStatus:
        self.submitted.append((symbol, side, qty, limit_price))
        return OrderStatus(
            id="o1",
            client_order_id=client_order_id,
            symbol=symbol,
            status="new",
            type="limit",
            limit_price=limit_price,
        )

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    async def order(self, order_id: str) -> OrderStatus:
        return OrderStatus(
            id=order_id, client_order_id="x", symbol="SPY", status=self.final, type="limit"
        )


async def _no_wait(_: float) -> None:
    return None


async def test_smoke_places_far_limit_and_confirms_cancel() -> None:
    broker = _Broker("canceled")
    result = await paper_smoke(
        cast(AlpacaPaperBroker, broker), _Bars(), SimClock(NOW), wait=_no_wait
    )
    assert broker.submitted == [("SPY", OrderSide.BUY, 1, 350.0)]
    assert broker.cancelled == ["o1"]
    assert result.ok
    assert result.statuses == ("new", "canceled")


async def test_smoke_fails_if_order_filled() -> None:
    broker = _Broker("filled")
    result = await paper_smoke(
        cast(AlpacaPaperBroker, broker), _Bars(), SimClock(NOW), wait=_no_wait
    )
    assert not result.ok
