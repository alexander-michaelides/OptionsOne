from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from typing import List, Optional
import requests
import urllib.parse
import dotenv
import websockets

HISTORICAL_NEWS_API_URL = "https://data.alpaca.markets/v1beta1/news"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

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
    
def ingest_from_alpaca(news_data: dict) -> NewsArticle:
    return NewsArticle(
        article_id=news_data['id'],
        headline=news_data['headline'],
        summary=news_data.get('summary') or '',
        content=news_data.get('content') or '',
        source=news_data.get('source') or news_data.get('author') or '',
        symbols=news_data.get('symbols') or [],
        published_at=datetime.fromisoformat(news_data['created_at']),
    )

def retrive_historical_news(start_date: datetime, end_date: datetime, limit: int) -> List[NewsArticle]:
    fields = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "limit": limit
    }
    
    query = HISTORICAL_NEWS_API_URL + "?" + urllib.parse.urlencode(fields)
    
    response = requests.get(query, headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    })
    
    return [ingest_from_alpaca(article) for article in response.json()]