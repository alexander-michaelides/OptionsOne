print(r"""
         _           _       _         _         _           _            _           _           _            _      
        /\ \        /\ \    /\ \      /\ \      /\ \        /\ \     _   / /\        /\ \        /\ \     _   /\ \    
       /  \ \      /  \ \   \_\ \     \ \ \    /  \ \      /  \ \   /\_\/ /  \      /  \ \      /  \ \   /\_\/  \ \   
      / /\ \ \    / /\ \ \  /\__ \    /\ \_\  / /\ \ \    / /\ \ \_/ / / / /\ \__  / /\ \ \    / /\ \ \_/ / / /\ \ \  
     / / /\ \ \  / / /\ \_\/ /_ \ \  / /\/_/ / / /\ \ \  / / /\ \___/ / / /\ \___\/ / /\ \ \  / / /\ \___/ / / /\ \_\ 
    / / /  \ \_\/ / /_/ / / / /\ \ \/ / /   / / /  \ \_\/ / /  \/____/\ \ \ \/___/ / /  \ \_\/ / /  \/____/ /_/_ \/_/ 
   / / /   / / / / /__\/ / / /  \/_/ / /   / / /   / / / / /    / / /  \ \ \    / / /   / / / / /    / / / /____/\    
  / / /   / / / / /_____/ / /     / / /   / / /   / / / / /    / / _    \ \ \  / / /   / / / / /    / / / /\____\/    
 / / /___/ / / / /     / / /  ___/ / /__ / / /___/ / / / /    / / /_/\__/ / / / / /___/ / / / /    / / / / /______    
/ / /____\/ / / /     /_/ /  /\__\/_/___/ / /____\/ / / /    / / /\ \/___/ / / / /____\/ / / /    / / / / /_______\   
\/_________/\/_/      \_\/   \/_________\/_________/\/_/     \/_/  \_____\/  \/_________/\/_/     \/_/\/__________/   
""")
import os
import warnings
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import market
from market import MarketContext
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings(
    "ignore",
    message="laya: this checkpoint ships invalid temperatures.*"
)

from laya import Router
import dotenv
dotenv.load_dotenv()

import news
from news import NewsArticle, NewsSignal

import agent
print("Initializing...")
router = Router(preload=True, token = os.getenv('HF_TOKEN'))
print("Router initialized.")

SIGNAL_LOG_PATH = Path(__file__).with_name("signal_history.jsonl")
OUTCOME_TASKS = set()
ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"

def process_article(router, article):
    signals = agent.analyze_article(router, article)
    contexts = {
        context.ticker: context
        for context in market.create_from_article(article)
    }
    return [
        (signal, contexts.get(signal.ticker))
        for signal in signals
    ]

def calculate_signal_quality(
    signal: NewsSignal,
    context: MarketContext
) -> float:
    """
    Returns signal quality from 0.0 to 1.0.

    Quality measures how strong/actionable the news signal appears.
    It does NOT determine bullish vs bearish direction.
    """

    # -------------------------
    # News quality
    # -------------------------

    news_quality = (
        0.30 * abs(signal.direction)
        + 0.25 * signal.materiality
        + 0.20 * signal.novelty
        + 0.10 * signal.market_relevance
        + 0.10 * signal.company_relevance
        + 0.05 * signal.certainty
    )

    # Penalize information Laya believes was already known
    news_quality *= (1.0 - signal.already_known)


    # -------------------------
    # Has the market already moved?
    # -------------------------

    move = abs(context.return_since_news)

    if move < 0.0025:        # < 0.25%
        freshness = 1.00

    elif move < 0.005:       # 0.25% - 0.50%
        freshness = 0.90

    elif move < 0.010:       # 0.50% - 1.00%
        freshness = 0.70

    elif move < 0.020:       # 1% - 2%
        freshness = 0.40

    else:                    # > 2%
        freshness = 0.15


    # -------------------------
    # Relative volume
    # -------------------------

    rv = context.relative_volume

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


    # -------------------------
    # Is recent movement aligned
    # with the news direction?
    # -------------------------

    direction = signal.direction

    if context.return_5m is None or not math.isfinite(context.return_5m):
        alignment = 0.50

    elif direction * context.return_5m > 0:
        alignment = 1.00

    elif abs(context.return_5m) < 0.001:
        alignment = 0.70

    else:
        alignment = 0.30


    # -------------------------
    # Combine everything
    # -------------------------

    quality = (
        0.65 * news_quality
        + 0.15 * freshness
        + 0.10 * volume_quality
        + 0.10 * alignment
    )

    return max(0.0, min(1.0, quality))


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _direction_label(direction: float) -> str:
    if direction > 0.15:
        return "UP"
    if direction < -0.15:
        return "DOWN"
    return "FLAT"


def _movement_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return f"{ANSI_YELLOW}n/a{ANSI_RESET}"
    color = ANSI_GREEN if value > 0 else ANSI_RED if value < 0 else ANSI_YELLOW
    arrow = "↑" if value > 0 else "↓" if value < 0 else "→"
    return f"{color}{arrow} {value:+.2%}{ANSI_RESET}"


def _expected_text(direction: float) -> str:
    label = _direction_label(direction)
    color = ANSI_GREEN if label == "UP" else ANSI_RED if label == "DOWN" else ANSI_YELLOW
    return f"{color}{label} ({direction:+.2f}){ANSI_RESET}"


def _quality_text(quality: float) -> str:
    color = ANSI_GREEN if quality >= 0.75 else ANSI_YELLOW if quality >= 0.50 else ANSI_RED
    return f"{color}{quality:.1%}{ANSI_RESET}"


def _store_event(event: dict):
    with SIGNAL_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def _signal_event(signal: NewsSignal, context: MarketContext, quality: float) -> dict:
    return {
        "type": "signal",
        "article_id": signal.article_id,
        "ticker": signal.ticker,
        "headline": getattr(signal, "headline", None),
        "published_at": None,
        "captured_at": context.captured_at.isoformat(),
        "expected_direction": signal.direction,
        "expected_label": _direction_label(signal.direction),
        "quality": quality,
        "event_type": signal.event_type,
        "effect_type": signal.effect_type,
        "time_horizon": signal.time_horizon,
        "positive_probability": signal.positive_probability,
        "neutral_probability": signal.neutral_probability,
        "negative_probability": signal.negative_probability,
        "certainty": signal.certainty,
        "materiality": signal.materiality,
        "novelty": signal.novelty,
        "company_relevance": signal.company_relevance,
        "market_relevance": signal.market_relevance,
        "already_known": signal.already_known,
        "price_at_news": _finite_or_none(context.price_at_news),
        "price_at_signal": _finite_or_none(context.price_at_signal),
        "movement_since_news": _finite_or_none(context.return_since_news),
        "volume": context.volume,
        "average_volume": context.avg_volume,
        "relative_volume": _finite_or_none(context.relative_volume),
        "return_1m": _finite_or_none(context.return_1m),
        "return_5m": _finite_or_none(context.return_5m),
        "realized_volatility": _finite_or_none(context.realized_volatility),
        "spy_return_5m": _finite_or_none(context.spy_return_5m),
    }


async def track_outcomes(signal: NewsSignal, context: MarketContext, quality: float):
    """Capture price movement one, five, and ten minutes after the signal."""
    last_elapsed = 0
    for minutes in (1, 5, 10):
        elapsed = minutes * 60
        await asyncio.sleep(elapsed - last_elapsed)
        last_elapsed = elapsed

        checked_at = datetime.now(timezone.utc)
        try:
            current_price = await asyncio.to_thread(
                market.get_price_at, signal.ticker, checked_at
            )
        except Exception as exc:
            print(f"{ANSI_YELLOW}{signal.ticker} +{minutes}m price lookup failed: {exc}{ANSI_RESET}")
            current_price = math.nan

        baseline = context.price_at_signal
        movement = current_price / baseline - 1 if (
            math.isfinite(current_price) and math.isfinite(baseline) and baseline != 0
        ) else math.nan
        since_news = current_price / context.price_at_news - 1 if (
            math.isfinite(current_price)
            and math.isfinite(context.price_at_news)
            and context.price_at_news != 0
        ) else math.nan
        aligned = (
            "aligned" if math.isfinite(movement) and signal.direction * movement > 0
            else "against expectation" if math.isfinite(movement) and signal.direction * movement < 0
            else "flat / unavailable"
        )

        outcome = {
            "type": "outcome",
            "article_id": signal.article_id,
            "ticker": signal.ticker,
            "horizon_minutes": minutes,
            "checked_at": checked_at.isoformat(),
            "expected_direction": signal.direction,
            "expected_label": _direction_label(signal.direction),
            "quality": quality,
            "price_at_signal": _finite_or_none(baseline),
            "price_at_check": _finite_or_none(current_price),
            "movement_since_signal": _finite_or_none(movement),
            "movement_since_news": _finite_or_none(since_news),
            "aligned_with_expectation": aligned,
        }
        _store_event(outcome)

        print(
            f"\n{ANSI_CYAN}{signal.ticker} +{minutes}m{ANSI_RESET}  "
            f"movement {_movement_text(_finite_or_none(movement))} since signal  |  "
            f"since news {_movement_text(_finite_or_none(since_news))}  |  "
            f"expected {_expected_text(signal.direction)}  |  "
            f"quality {_quality_text(quality)}  |  {aligned}"
        )


async def run_news_stream():
    print("Connecting to Alpaca news stream...")
    async for article in news.stream_news():
        print(f"\nNews: {article.headline} ({', '.join(article.symbols)})")
        for signal, context in process_article(router, article):
            if context is not None:
                quality = calculate_signal_quality(signal, context)
                initial = _signal_event(signal, context, quality)
                initial["headline"] = article.headline
                initial["published_at"] = article.published_at.isoformat()
                _store_event(initial)

                print(
                    f"{ANSI_CYAN}{signal.ticker}{ANSI_RESET}  "
                    f"movement since news {_movement_text(_finite_or_none(context.return_since_news))}  |  "
                    f"expected {_expected_text(signal.direction)}  |  "
                    f"quality {_quality_text(quality)}"
                )
                task = asyncio.create_task(track_outcomes(signal, context, quality))
                OUTCOME_TASKS.add(task)
                task.add_done_callback(OUTCOME_TASKS.discard)


if __name__ == "__main__":
    try:
        asyncio.run(run_news_stream())
    except KeyboardInterrupt:
        print("News stream stopped.")
