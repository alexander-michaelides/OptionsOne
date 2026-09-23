from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

@dataclass
class MarketContext:
    article_id: int
    ticker: str
    
    captured_at: datetime = field(default_factory=datetime.now(timezone.utc))
    
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
    