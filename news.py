# handles news streaming and processing

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import json
import os
import time
from typing import AsyncIterator, List
import requests
from requests.exceptions import RequestException
import dotenv
import websockets

HISTORICAL_NEWS_API_URL = "https://data.alpaca.markets/v1beta1/news"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
HISTORICAL_RECEIVED_DELAY = timedelta(seconds=30)
HISTORICAL_API_ATTEMPTS = 5


def _get_historical_page(params: dict, headers: dict):
    """Retry transient HTTP failures and Alpaca rate limits."""
    for attempt in range(HISTORICAL_API_ATTEMPTS):
        try:
            response = requests.get(
                HISTORICAL_NEWS_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
        except RequestException:
            if attempt == HISTORICAL_API_ATTEMPTS - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == HISTORICAL_API_ATTEMPTS - 1:
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(delay, 60))
            continue

        return response

dotenv.load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_SECRET_KEY")

@dataclass
class NewsArticle:
    article_id: int
    
    headline: str
    summary: str
    content: str
    
    source: str
    
    symbols: List[str]
    
    published_at: datetime
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class NewsSignal:
    article_id: int
    ticker: str
    
    event_type: str
    effect_type: str
    time_horizon: str
    
    direction: float
    
    positive_probability: float
    neutral_probability: float
    negative_probability: float

    certainty: float
    materiality: float
    novelty: float
    
    company_relevance: float
    market_relevance: float
    already_known: float

    # Timing
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self) -> str:
        return (
            f"\n"
            f"╔══════════════════════════════════════╗\n"
            f"║          NEWS SIGNAL — {self.ticker:<12}║\n"
            f"╠══════════════════════════════════════╣\n"
            f"║ Event       │ {self.event_type:<22}║\n"
            f"║ Effect      │ {self.effect_type:<22}║\n"
            f"║ Horizon     │ {self.time_horizon:<22}║\n"
            f"╠══════════════════════════════════════╣\n"
            f"║ Direction   │ {self.direction:>+6.2f}                ║\n"
            f"║   Positive  │ {self.positive_probability:>6.1%}                ║\n"
            f"║   Neutral   │ {self.neutral_probability:>6.1%}                ║\n"
            f"║   Negative  │ {self.negative_probability:>6.1%}                ║\n"
            f"╠══════════════════════════════════════╣\n"
            f"║ Materiality │ {self.materiality:>6.1%}                ║\n"
            f"║ Novelty     │ {self.novelty:>6.1%}                ║\n"
            f"║ Certainty   │ {self.certainty:>6.1%}                ║\n"
            f"╠══════════════════════════════════════╣\n"
            f"║ Company Rel │ {self.company_relevance:>6.1%}                ║\n"
            f"║ Market Rel  │ {self.market_relevance:>6.1%}                ║\n"
            f"║ Known       │ {self.already_known:>6.1%}                ║\n"
            f"╚══════════════════════════════════════╝"
        )
    
def ingest_from_alpaca(news_data: dict, received_at: datetime) -> NewsArticle:
    return NewsArticle(
        article_id=news_data['id'],
        headline=news_data['headline'],
        summary=news_data.get('summary') or '',
        content=news_data.get('content') or '',
        source=news_data.get('source') or news_data.get('author') or '',
        symbols=news_data.get('symbols') or [],
        published_at=datetime.fromisoformat(news_data.get('created_at')),
        received_at=received_at,
    )


async def stream_news():
    """Yield NewsArticle objects from Alpaca's live news WebSocket stream."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to stream news")

    async with websockets.connect(NEWS_STREAM_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "action": "auth",
            "key": ALPACA_API_KEY,
            "secret": ALPACA_API_SECRET,
        }))

        authenticated = False
        while not authenticated:
            response = json.loads(await ws.recv())
            messages = response if isinstance(response, list) else [response]
            for message in messages:
                if message.get("T") == "error":
                    raise RuntimeError(f"Alpaca news stream error: {message}")
                if message.get("T") == "success" and message.get("msg") == "authenticated":
                    authenticated = True

        await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))

        async for raw_message in ws:
            events = json.loads(raw_message)
            if not isinstance(events, list):
                events = [events]
            for event in events:
                if event.get("T") == "error":
                    raise RuntimeError(f"Alpaca news stream error: {event}")
                if event.get("T") == "n":
                    yield ingest_from_alpaca(event)

async def stream_historical_news(
    start_date: datetime,
    end_date: datetime,
) -> AsyncIterator[NewsArticle]:

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "to retrieve historical news"
        )

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }

    page_token = None

    while True:

        params = {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "limit": 50,
            "sort": "asc",
        }

        if page_token is not None:
            params["page_token"] = page_token

        response = await asyncio.to_thread(
            _get_historical_page,
            params,
            headers,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Failed to retrieve historical news: "
                f"{response.status_code} - {response.text}"
            )

        payload = response.json()

        articles = payload.get("news", [])

        for raw_article in articles:

            published_at = datetime.fromisoformat(
                raw_article["created_at"]
            )

            received_at = (
                published_at
                + HISTORICAL_RECEIVED_DELAY
            )

            yield ingest_from_alpaca(
                raw_article,
                received_at=received_at,
            )

        page_token = payload.get("next_page_token")

        if not page_token:
            break
