from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

@dataclass
class NewsArticle:
    article_id: int
    
    headline: str
    summary: str
    content: str
    
    source: str
    
    symbols: List[str]
    
    published_at: datetime
    received_at: datetime = field(default_factory=datetime.now(timezone.utc))

@dataclass
class NewsSignal:
    article_id: int
    ticker: str
    
    # Event Information
    sentiment: float
    confidence: float
    impact: float
    novelty: float
    event_type: str
    
    # Timing
    analyzed_at: datetime = field(default_factory=datetime.now(timezone.utc))
    
def ingest_from_alpaca(news_data: dict) -> NewsArticle:
    return NewsArticle(
        article_id=news_data['id'],
        headline=news_data['headline'],
        summary=news_data['summary'],
        content=news_data['content'],
        source=news_data['source'],
        symbols=news_data['symbols'],
        published_at=datetime.fromisoformat(news_data['published_at']),
    )