from datetime import UTC, datetime, timedelta
from math import sqrt

import pytest

from optionsone.backtest.portfolio import Trade
from optionsone.backtest.report import EquityPoint, build_report


def test_report_uses_final_daily_equity_and_intraday_drawdown() -> None:
    day1 = datetime(2026, 1, 5, 16, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    curve = [
        EquityPoint(day1, 110),
        EquityPoint(day2, 99),
        EquityPoint(day2 + timedelta(hours=1), 115.5),
    ]
    trades = (Trade("A", 1, day1, day1, 100, 10, 10), Trade("A", 1, day2, day2, 110, 5.5, 5.5))

    report = build_report(100, curve, trades, ())

    # Both series are [.10, .05]. Sample standard deviation is .05 / sqrt(2).
    assert report.sharpe_per_trade == pytest.approx(0.075 / (0.05 / sqrt(2)))
    assert report.sharpe_per_day == pytest.approx(sqrt(252) * report.sharpe_per_trade)
    assert report.max_drawdown == pytest.approx(0.1)
