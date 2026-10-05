from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import news
from storage import Storage

class SessionType(Enum):
    LIVE = "live"
    HISTORICAL = "historical"

class SessionStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SessionConfig:
    name: str
    type: SessionType

    start: datetime | None = None
    end: datetime | None = None

    weights: dict[str, float] = field(default_factory=dict)


class Session:
    def __init__(
        self,
        config: SessionConfig,
        storage: Storage,
    ):
        self.id = str(uuid4())
        self.config = config
        self.storage = storage

        self.status = SessionStatus.CREATED

        self.created_at = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

        # for historical sessions
        self._current_time: datetime | None = config.start

    # function to have other modules blind to whether session is historical
    def now(self) -> datetime:
        if self.config.type == SessionType.LIVE:
            return datetime.now(timezone.utc)

        if self._current_time is None:
            raise RuntimeError("Session has no current time")

        return self._current_time

    def set_time(self, time: datetime):
        if self.config.type == SessionType.LIVE:
            raise RuntimeError("Cannot set time on live session")

        if (
            self._current_time is not None
            and time < self._current_time
        ):
            raise ValueError("Cannot set time backwards")

        self._current_time = time

    async def news(self):
        if self.config.type == SessionType.LIVE:
            async for article in news.stream_news():
                yield article
        else:
            if self.config.start is None or self.config.end is None:
                raise ValueError("Session has no start or end time")

            async for article in news.stream_historical_news(
                    self.config.start, self.config.end
            ):
                if self._current_time is None or article.published_at > self._current_time:
                    self.set_time(article.published_at)

                yield article

    def start(self):
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

        self.storage.save_session(self)

    def complete(self):
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now(timezone.utc)

        self.storage.save_session(self)

    def fail(self):
        self.status = SessionStatus.FAILED
        self.ended_at = datetime.now(timezone.utc)

        self.storage.save_session(self)