# handles all aspects of market data, price, returns, etc

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import os

import dotenv
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame

from news import NewsArticle


dotenv.load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to retrieve market data")

_DATA_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

_SPY = "SPY"


@dataclass
class MarketContext:
    article_id: int
    ticker: str

    price_at_news: float
    price_at_signal: float

    return_since_news: float

    volume: int
    avg_volume: int
    relative_volume: float

    return_1m: float
    return_5m: float

    realized_volatility: float | None

    spy_return_5m: float | None

    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def _number(value: float | None, formatter: str) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return format(value, formatter)

    def __str__(self) -> str:
        rows = [
            ("Article", str(self.article_id)),
            ("Ticker", self.ticker),
            ("Price at news", self._number(self.price_at_news, ".2f")),
            ("Price at signal", self._number(self.price_at_signal, ".2f")),
            ("Return since news", self._number(self.return_since_news, ".2%")),
            ("Volume", f"{self.volume:,}"),
            ("Average volume", f"{self.avg_volume:,}"),
            ("Relative volume", f"{self._number(self.relative_volume, '.2f')}x"
             if math.isfinite(self.relative_volume) else "n/a"),
            ("Return (1m)", self._number(self.return_1m, ".2%")),
            ("Return (5m)", self._number(self.return_5m, ".2%")),
            ("Realized volatility", self._number(self.realized_volatility, ".2%")),
            ("SPY return (5m)", self._number(self.spy_return_5m, ".2%")),
            ("Captured at", self.captured_at.isoformat(timespec="seconds")),
        ]
        label_width = max(len(label) for label, _ in rows)
        value_width = max(len(value) for _, value in rows)
        inner_width = label_width + value_width + 5
        lines = [f"┌{'─' * inner_width}┐", f"│ {'MARKET CONTEXT':^{inner_width - 2}} │", f"├{'─' * inner_width}┤"]
        lines.extend(
            f"│ {label:<{label_width}}  {value:>{value_width}} │"
            for label, value in rows
        )
        lines.append(f"└{'─' * inner_width}┘")
        return "\n".join(lines)

@dataclass
class MarketOutcome:
    article_id: int
    ticker: str

    starting_price: float

    return_1m: float | None
    return_5m: float | None
    return_15m: float | None
    return_30m: float | None
    return_60m: float | None

    measured_from: datetime

def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def get_minute_bars(ticker: str, start: datetime, end: datetime) -> list:
    """Fetch sorted IEX one-minute bars for a ticker and time range."""
    response = _DATA_CLIENT.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=_utc(start),
        end=_utc(end),
        feed=DataFeed.IEX,
        limit=10000,
    ))
    try:
        bars = list(response[ticker])
    except (KeyError, TypeError):
        bars = []
    return sorted(
        bars,
        key=lambda bar: _utc(bar.timestamp)
    )

def get_price_at(
        ticker: str,
        timestamp: datetime
) -> float:
    timestamp = _utc(timestamp)

    bars = get_minute_bars(
        ticker,
        timestamp - timedelta(days=7),
        timestamp
    )

    if not bars:
        return math.nan

    return float(
        bars[-1].close
    )


def _pct_return(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old) or old == 0:
        return math.nan
    return new / old - 1.0


def _recent_metrics(bars):
    if not bars:
        return math.nan, math.nan, math.nan, 0, 0, math.nan, None

    closes = [
        float(bar.close)
        for bar in bars
    ]

    volumes = [
        int(bar.volume)
        for bar in bars
    ]

    price = closes[-1]
    volume = volumes[-1]

    # previous 20 minute bars
    prior_volumes = volumes[-21:-1]

    if prior_volumes:
        avg_volume = int(round(sum(prior_volumes) / len(prior_volumes)))
    else:
        avg_volume = 0

    relative_volume = volume / avg_volume if avg_volume else math.nan

    return_1m = _pct_return(price, closes[-2]) if len(closes) >= 2 else math.nan
    return_5m = _pct_return(price, closes[-6]) if len(closes) >= 6 else math.nan

    #stdev of most recent 30 min log returns
    recent_closes = closes[-31:]

    log_returns = [
        math.log(new / old)
        for old, new in zip(
            recent_closes,
            recent_closes[1:]
        )

        if old > 0 and new > 0
    ]

    volatility = None

    if len(log_returns) >= 2:
        mean = sum(log_returns) / len(log_returns)
        variance = sum(
            (value - mean) ** 2
            for value in log_returns
        ) / (len(log_returns) - 1)

        volatility = math.sqrt(variance)

    return price, return_1m, return_5m, volume, avg_volume, relative_volume, volatility



def create_from_article(article: NewsArticle, current_time: datetime) -> list[MarketContext]:
    published_at = _utc(article.published_at)
    captured_at = _utc(current_time)

    contexts: list[MarketContext] = []

    spy_return_5m = None

    try:
        spy_bars = get_minute_bars(
            _SPY, captured_at - timedelta(minutes=10), captured_at
        )
        (
            _,
            _,
            spy_return_5m,
            _,
            _,
            _,
            _,
        ) = _recent_metrics(spy_bars)
        if not math.isfinite(spy_return_5m):
            spy_return_5m = None
    except Exception:
        spy_return_5m = None

    for ticker in dict.fromkeys(article.symbols):
        try:
            price_at_news = get_price_at(
                ticker,
                published_at
            )
            signal_bars = get_minute_bars(
                ticker,
                captured_at - timedelta(minutes=40),
                captured_at
            )
        except APIError as exc:
            if "invalid symbol" not in str(exc).lower():
                raise
            print(f"Skipping market data for unsupported Alpaca symbol {ticker}: {exc}")
            continue

        (price_at_signal,
         return_1m,
         return_5m,
         volume,
         avg_volume,
         relative_volume,
         realized_volatility) = _recent_metrics(signal_bars)

        context = MarketContext(
            article_id=article.article_id,
            ticker=ticker,
            price_at_news=price_at_news,
            price_at_signal=price_at_signal,
            return_since_news=_pct_return(price_at_signal, price_at_news),
            volume=volume,
            avg_volume=avg_volume,
            relative_volume=relative_volume,
            return_1m=return_1m,
            return_5m=return_5m,
            realized_volatility=realized_volatility,
            spy_return_5m=spy_return_5m,
            captured_at=captured_at,
        )
        contexts.append(context)

    return contexts

def create_outcome(
    article_id: int,
    ticker: str,
    current_time: datetime,
) -> MarketOutcome:
    #only used for historical evaluation

    current_time = _utc(current_time)

    # calculates every outcome in one request
    bars = get_minute_bars(
        ticker,
        current_time - timedelta(minutes=2),
        current_time + timedelta(minutes=61),
    )

    starting_bars = [
        bar
        for bar in bars
        if _utc(bar.timestamp) <= current_time
    ]

    if starting_bars:
        starting_price = float(
            starting_bars[-1].close
        )
    else:
        starting_price = math.nan

    def future_price_at(
        minutes: int,
    ) -> float | None:

        target = (
            current_time
            + timedelta(minutes=minutes)
        )

        for bar in bars:

            bar_time = _utc(
                bar.timestamp
            )

            if bar_time >= target:

                difference = (
                    bar_time - target
                )

                if difference > timedelta(minutes=1):
                    return None

                return float(
                    bar.close
                )

        return None

    def future_return(
        minutes: int,
    ) -> float | None:

        if not math.isfinite(
            starting_price
        ):
            return None

        future_price = future_price_at(
            minutes
        )

        if future_price is None:
            return None

        result = _pct_return(
            future_price,
            starting_price,
        )

        if not math.isfinite(result):
            return None

        return result

    return MarketOutcome(
        article_id=article_id,
        ticker=ticker,

        starting_price=starting_price,

        return_1m=future_return(1),
        return_5m=future_return(5),
        return_15m=future_return(15),
        return_30m=future_return(30),
        return_60m=future_return(60),

        measured_from=current_time,
    )