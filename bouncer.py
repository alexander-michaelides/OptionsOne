# handles the scoring of each signal

import math

from news import NewsSignal
from market import MarketContext

def score(news: NewsSignal, market: MarketContext) -> float:

    # general quality weights based on news
    # TODO: backtest/train weights
    news_quality = (
            0.30 * abs(news.direction)
            + 0.25 * news.materiality
            + 0.20 * news.novelty
            + 0.10 * news.market_relevance
            + 0.10 * news.company_relevance
            + 0.05 * news.certainty
    )

    # penalize news that agent believes was already known
    news_quality *= (1.0 - news.already_known)

    # how much the market has moved since the news, no use in trading off of a signal that already moved
    move = abs(market.return_since_news)

    if move < 0.0025:  # < 0.25%
        freshness = 1.00

    elif move < 0.005:  # 0.25% - 0.50%
        freshness = 0.90

    elif move < 0.010:  # 0.50% - 1.00%
        freshness = 0.70

    elif move < 0.020:  # 1% - 2%
        freshness = 0.40

    else:  # > 2%
        freshness = 0.15

    # relative market volume

    rv = market.relative_volume

    if rv >= 2.0:
        volume_quality = 1.00

    elif rv >= 1.5:
        volume_quality = 0.90

    elif rv >= 1.0:
        volume_quality = 0.75

    elif rv >= 0.5:
        volume_quality = 0.50

    else:
        volume_quality = 0.25


    # checking if the direction predicted by agent is aligned with the stock direction
    direction = news.direction

    if market.return_5m is None or not math.isfinite(market.return_5m):
        alignment = 0.50

    elif direction * market.return_5m > 0:
        alignment = 1.00

    elif abs(market.return_5m) < 0.001:
        alignment = 0.70

    else:
        alignment = 0.30

    # combine market quality and news quality
    quality = (
            0.65 * news_quality
            + 0.15 * freshness
            + 0.10 * volume_quality
            + 0.10 * alignment
    )

    return max(0.0, min(1.0, quality))