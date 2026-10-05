# handles all aspects of market data, price, returns, etc

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import os

import dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
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


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)

def _fetch_minute_bars(ticker: str, start: datetime, end: datetime):
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        limit=10000,
    )
    return _DATA_CLIENT.get_stock_bars(request)


def _extract_bars(response, ticker: str):
    """Normalize Alpaca BarSet, mapping, and iterable response shapes."""
    try:
        values = response[ticker]
    except (KeyError, TypeError):
        values = response

    def collect(value):
        if hasattr(value, "timestamp") and hasattr(value, "close"):
            return [value]
        if isinstance(value, dict):
            result = []
            for child in value.values():
                result.extend(collect(child))
            return result
        if isinstance(value, (tuple, list)):
            result = []
            for child in value:
                result.extend(collect(child))
            return result
        return []

    return collect(values)


def _price_at_from_bars(bars, timestamp: datetime) -> float:
    eligible = [bar for bar in bars if _utc(bar.timestamp) <= timestamp]
    return float(eligible[-1].close) if eligible else math.nan


def get_price_at(ticker: str, timestamp: datetime) -> float:
    timestamp = _utc(timestamp)
    response = _fetch_minute_bars(ticker, timestamp - timedelta(days=7), timestamp)
    return _price_at_from_bars(_extract_bars(response, ticker), timestamp)


def _pct_return(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old) or old == 0:
        return math.nan
    return new / old - 1.0


def _recent_metrics(bars):
    if not bars:
        return math.nan, math.nan, math.nan, 0, 0, math.nan, None

    closes = [float(bar.close) for bar in bars]
    volumes = [int(bar.volume) for bar in bars]
    price = closes[-1]
    volume = volumes[-1]
    prior_volumes = volumes[-21:-1]
    avg_volume = int(round(sum(prior_volumes) / len(prior_volumes))) if prior_volumes else 0
    relative_volume = volume / avg_volume if avg_volume else math.nan

    return_1m = _pct_return(price, closes[-2]) if len(closes) >= 2 else math.nan
    return_5m = _pct_return(price, closes[-6]) if len(closes) >= 6 else math.nan

    # Standard deviation of the most recent 30 one-minute log returns.
    returns = [math.log(b / a) for a, b in zip(closes[-31:-1], closes[-30:]) if a > 0 and b > 0]
    volatility = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        volatility = math.sqrt(sum((value - mean) ** 2 for value in returns) / (len(returns) - 1))

    return price, return_1m, return_5m, volume, avg_volume, relative_volume, volatility


def create_from_article(article: NewsArticle, historical:bool=False) -> list[MarketContext]:
    published_at = _utc(article.published_at)

    if historical:
        captured_at = published_at + timedelta(milliseconds=10)
    else:
        captured_at = datetime.now(timezone.utc)

    mcs: list[MarketContext] = []

    for ticker in dict.fromkeys(article.symbols):
        news_bars = _extract_bars(
            _fetch_minute_bars(ticker, published_at - timedelta(days=7), published_at), ticker
        )
        signal_bars = _extract_bars(
            _fetch_minute_bars(ticker, captured_at - timedelta(minutes=40), captured_at), ticker
        )
        price_at_news = _price_at_from_bars(news_bars, published_at)
        (price_at_signal, return_1m, return_5m, volume, avg_volume,
         relative_volume, realized_volatility) = _recent_metrics(signal_bars)

        spy_return_5m = None
        try:
            spy_bars = _extract_bars(
                _fetch_minute_bars(_SPY, captured_at - timedelta(minutes=10), captured_at), _SPY
            )
            _, _, spy_return_5m, _, _, _, _ = _recent_metrics(spy_bars)
            if not math.isfinite(spy_return_5m):
                spy_return_5m = None
        except Exception:
            spy_return_5m = None

        mcs.append(MarketContext(
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
        ))

    return mcs
