from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from optionsone.core.types import OrderIntent, OrderSide, Side, Signal
from optionsone.risk.kill_switch import KillSwitch, KillSwitchConfig
from optionsone.risk.limits import RiskLimits, pre_trade_check
from optionsone.risk.sizing import SizingConfig, size_shares

NOW = datetime(2026, 1, 5, 15, tzinfo=UTC)
ORDER = OrderIntent(client_order_id="entry", symbol="A", side=OrderSide.BUY, qty=10, created_at=NOW)


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (RiskLimits(max_position_notional_pct=0.09), "max_position_notional_pct"),
        (RiskLimits(max_gross_exposure=0.19), "max_gross_exposure"),
        (RiskLimits(max_open_positions=1), "max_open_positions"),
        (RiskLimits(max_orders_per_min=1), "max_orders_per_min"),
    ],
)
def test_limits_deny_projected_over_cap_orders(limits: RiskLimits, reason: str) -> None:
    decision = pre_trade_check(
        ORDER,
        limits,
        equity=10_000,
        quantities={"B": -10},
        prices={"A": 100, "B": 100},
        submitted_at=[NOW - timedelta(seconds=59)],
    )

    assert not decision.allowed
    assert decision.reason == reason


def test_limits_allow_safe_exit_even_when_caps_are_breached() -> None:
    exit_order = ORDER.model_copy(update={"side": OrderSide.SELL})

    decision = pre_trade_check(
        exit_order,
        RiskLimits(max_orders_per_min=1),
        equity=-1,
        quantities={"A": 10},
        prices={},
        submitted_at=[NOW],
    )

    assert decision.allowed


@pytest.mark.parametrize(
    ("daily", "drawdown", "peak", "equity", "reason"),
    [(0.05, 0.20, 100, 94, "daily_loss"), (0.10, 0.05, 110, 104, "peak_drawdown")],
)
def test_kill_switch_latches_until_authenticated_reset(
    daily: float, drawdown: float, peak: float, equity: float, reason: str
) -> None:
    switch = KillSwitch(
        KillSwitchConfig(
            max_daily_loss_pct=daily, max_drawdown_pct=drawdown, reset_code=SecretStr("approve")
        )
    )
    session = NOW.date()
    switch.observe(session, 100)
    switch.observe(session, peak)
    switch.observe(session, equity)
    switch.observe(session + timedelta(days=1), 150)

    assert switch.reason == reason
    assert switch.flatten_requested
    assert not switch.allows(ORDER, 0)
    assert not switch.allows(ORDER, -9)
    assert switch.allows(ORDER, -10)
    assert not switch.reset("wrong")
    assert not switch.allows(ORDER, 0)
    assert switch.reset("approve")
    assert not switch.flatten_requested
    assert switch.allows(ORDER, 0)


@pytest.mark.parametrize(("vol", "expected"), [(0.02, 25), (0.04, 12), (0.0, 30)])
def test_sizing_scales_with_volatility_and_respects_caps(vol: float, expected: int) -> None:
    signal = Signal(
        article_id=1,
        ticker="A",
        as_of=NOW,
        side=Side.LONG,
        strength=0.5,
        horizon=timedelta(minutes=5),
        model_id="test",
    )

    assert (
        size_shares(signal, 10_000, 10, vol, SizingConfig(target_bar_vol=0.001, max_shares=30))
        == expected
    )
