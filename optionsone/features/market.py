"""Market features over completed one-minute bars."""

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from math import log, sqrt
from statistics import fmean

from optionsone.core.types import Bar

MARKET_FEATURE_NAMES = (
    "return_since_news",
    "relative_volume",
    "return_1m",
    "return_5m",
    "realized_volatility",
    "spy_return_5m",
    "market_adjusted_return",
)


def market_features(
    stock: Sequence[Bar], benchmark: Sequence[Bar], news_time: datetime
) -> dict[str, float] | None:
    """Require 31 stock closes and exact 1m/5m endpoints at the latest stock close.

    Realised volatility is the root sum of squared log returns over the last 30
    observed changes. Relative volume uses the preceding 20 observations and
    requires a positive mean. Since-news returns use the last common completed
    close at or before classification availability as their baseline.
    """
    if len(stock) < 31:
        return None
    last = stock[-1]
    stock_closes = {bar.end: bar.close for bar in stock}
    benchmark_closes = {bar.end: bar.close for bar in benchmark}
    one_minute_ago = last.end - timedelta(minutes=1)
    five_minutes_ago = last.end - timedelta(minutes=5)
    if (
        one_minute_ago not in stock_closes
        or five_minutes_ago not in stock_closes
        or five_minutes_ago not in benchmark_closes
        or last.end not in benchmark_closes
    ):
        return None
    ends = [bar.end for bar in stock]
    baseline = next(
        (end for end in reversed(ends[: bisect_right(ends, news_time)]) if end in benchmark_closes),
        None,
    )
    avg_volume = fmean(bar.volume for bar in stock[-21:-1])
    if baseline is None or avg_volume == 0:
        return None
    recent = stock[-31:]
    since_news = last.close / stock_closes[baseline] - 1
    benchmark_return = benchmark_closes[last.end] / benchmark_closes[baseline] - 1
    return {
        "return_since_news": since_news,
        "relative_volume": last.volume / avg_volume,
        "return_1m": last.close / stock_closes[one_minute_ago] - 1,
        "return_5m": last.close / stock_closes[five_minutes_ago] - 1,
        "realized_volatility": sqrt(
            sum(log(after.close / before.close) ** 2 for before, after in pairwise(recent))
        ),
        "spy_return_5m": benchmark_closes[last.end] / benchmark_closes[five_minutes_ago] - 1,
        "market_adjusted_return": since_news - benchmark_return,
    }
