from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import json

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
        self._resume_after: tuple[datetime, int] | None = None
        self.articles_processed = 0

    @classmethod
    def resume(cls, session_id: str, storage: Storage) -> "Session":
        row = storage.get_session(session_id)
        if row is None:
            raise ValueError(f"No session found with id {session_id}")
        if row["type"] != SessionType.HISTORICAL.value:
            raise ValueError("Only historical sessions can be resumed")
        if row["status"] == SessionStatus.COMPLETED.value:
            raise ValueError("Cannot resume a completed session")

        saved = json.loads(row["config_json"])

        def parse_time(value):
            return datetime.fromisoformat(value) if value else None

        config = SessionConfig(
            name=saved["name"],
            type=SessionType(saved["type"]),
            start=parse_time(saved.get("start")),
            end=parse_time(saved.get("end")),
            weights=saved.get("weights", {}),
        )
        session = cls(config, storage)
        session.id = row["id"]
        session.status = SessionStatus(row["status"])
        session.created_at = parse_time(row["created_at"])
        session.started_at = parse_time(row["started_at"])
        session.ended_at = parse_time(row["ended_at"])

        progress = storage.get_session_progress(session_id)
        if progress is not None:
            published_at = parse_time(progress["published_at"])
            session._resume_after = (published_at, progress["article_id"])
            session._current_time = published_at
            session.articles_processed = progress["articles_processed"]
        return session

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

            resume_start = (
                self._resume_after[0]
                if self._resume_after is not None
                else self.config.start
            )
            async for article in news.stream_historical_news(
                    resume_start, self.config.end
            ):
                if self._resume_after is not None:
                    last_time, last_id = self._resume_after
                    if (article.published_at, article.article_id) <= (last_time, last_id):
                        continue

                if self._current_time is None or article.published_at > self._current_time:
                    self.set_time(article.published_at)

                yield article

    def mark_processed(self, article):
        self._resume_after = (article.published_at, article.article_id)
        self.articles_processed = self.storage.save_session_progress(
            self.id,
            article.article_id,
            article.published_at,
        )

    def progress(self) -> float:
        if self.config.start is None or self.config.end is None:
            return 0.0
        duration = (self.config.end - self.config.start).total_seconds()
        if duration <= 0:
            return 1.0
        elapsed = (self.now() - self.config.start).total_seconds()
        return min(1.0, max(0.0, elapsed / duration))

    def start(self):
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.ended_at = None

        self.storage.save_session(self)

    def complete(self):
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now(timezone.utc)

        self.storage.save_session(self)

    def fail(self):
        self.status = SessionStatus.FAILED
        self.ended_at = datetime.now(timezone.utc)

        self.storage.save_session(self)
