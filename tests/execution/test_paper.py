import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr

from optionsone.core.clock import SimClock
from optionsone.core.types import OrderIntent, OrderSide, Position
from optionsone.execution.alpaca_paper import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    AlpacaPaperSettings,
    LiveTradingForbidden,
)

NOW = datetime(2026, 1, 5, 15, tzinfo=UTC)
ORDERS = f"{PAPER_BASE_URL}/v2/orders"
POSITIONS = f"{PAPER_BASE_URL}/v2/positions"
ORDER = OrderIntent(
    client_order_id="signal-id", symbol="A", side=OrderSide.BUY, qty=10, created_at=NOW
)
REMOTE = {"client_order_id": "signal-id", "id": "remote-id"}


@pytest.fixture
def broker() -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        settings=AlpacaPaperSettings(
            base_url=PAPER_BASE_URL, api_key=SecretStr("fake"), secret_key=SecretStr("fake")
        ),
        clock=SimClock(NOW),
    )


@pytest.mark.parametrize("url", ["https://api.alpaca.markets", f"{PAPER_BASE_URL}/"])
def test_constructor_rejects_every_nonexact_paper_origin(url: str) -> None:
    with pytest.raises(LiveTradingForbidden):
        AlpacaPaperBroker(url)


def test_environment_cannot_override_paper_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_BASE_URL", "https://api.alpaca.markets")

    with pytest.raises(LiveTradingForbidden):
        AlpacaPaperBroker(PAPER_BASE_URL)


async def test_idempotent_resubmit_sends_one_market_order(broker: AlpacaPaperBroker) -> None:
    def accept(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "client_order_id": "signal-id",
            "symbol": "A",
            "side": "buy",
            "qty": "10",
            "type": "market",
            "time_in_force": "day",
        }
        return httpx.Response(201, json=REMOTE)

    with respx.mock as mock:
        mock.get(ORDERS).respond(json=[])
        mock.get(POSITIONS).respond(json=[])
        posted = mock.post(ORDERS).mock(side_effect=accept)

        await broker.submit(ORDER)
        await broker.submit(ORDER)

        assert posted.call_count == 1


async def test_startup_reconciliation_skips_known_order_ids(broker: AlpacaPaperBroker) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(ORDERS, params={"status": "open", "limit": "500", "direction": "desc"}).respond(
            json=[REMOTE]
        )
        mock.get(POSITIONS).respond(json=[{"symbol": "A", "qty": "5", "avg_entry_price": "99"}])
        posted = mock.post(ORDERS).respond(status_code=201, json=REMOTE)

        positions = await broker.reconcile()
        await broker.submit(ORDER)

        assert positions == (Position(symbol="A", qty=5, avg_price=99),)
        assert posted.call_count == 0


async def test_paper_fills_deliver_partials_once_and_cancels_at_paper_host(
    broker: AlpacaPaperBroker,
) -> None:
    activity_url = f"{PAPER_BASE_URL}/v2/account/activities"
    params = {
        "activity_types": "FILL",
        "direction": "asc",
        "page_size": "100",
        "after": "2026-01-05T15:00:00Z",
    }
    with respx.mock as mock:
        mock.get(ORDERS).respond(json=[REMOTE])
        mock.get(POSITIONS).respond(json=[])
        mock.get(activity_url, params={**params, "page_token": "activity-id"}).respond(json=[])
        mock.get(activity_url, params=params).respond(
            json=[
                {
                    "id": "activity-id",
                    "order_id": "remote-id",
                    "symbol": "A",
                    "side": "buy",
                    "qty": "3",
                    "price": "100.125",
                    "transaction_time": NOW.isoformat(),
                }
            ]
        )
        account_wide = mock.delete(ORDERS).respond(status_code=207, json=[])
        canceled = mock.delete(f"{ORDERS}/remote-id").respond(status_code=204)

        fills = await broker.fills()
        followup = await broker.fills()
        await broker.cancel(["signal-id"])

        assert [(fill.client_order_id, fill.qty, fill.price) for fill in fills] == [
            ("signal-id", 3, 100.125)
        ]
        assert not followup
        assert canceled.call_count == 1
        assert account_wide.call_count == 0  # never cancels foreign orders


async def test_paper_credentials_are_never_forwarded_to_redirect(
    broker: AlpacaPaperBroker,
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(ORDERS).respond(status_code=307, headers={"Location": "https://example.com"})
        redirected = mock.get("https://example.com").respond(json=[])

        with pytest.raises(httpx.HTTPStatusError):
            await broker.reconcile()

        assert redirected.call_count == 0


def test_live_account_key_is_refused() -> None:
    from optionsone.execution.alpaca_paper import LiveKeyForbidden

    settings = AlpacaPaperSettings(api_key=SecretStr("AKLIVE123"), secret_key=SecretStr("s"))
    with pytest.raises(LiveKeyForbidden):
        AlpacaPaperBroker(settings=settings)


def test_generic_alpaca_env_names_are_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    settings = AlpacaPaperSettings()
    assert settings.api_key.get_secret_value() == "PKTEST"
