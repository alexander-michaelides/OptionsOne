import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from news import NewsArticle, NewsSignal
from market import MarketContext

if TYPE_CHECKING:
    from session import Session

DEFAULT_DB_PATH = "data/trading.db"

class Storage:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(path)

        self.connection.row_factory = sqlite3.Row

        self.connection.execute("PRAGMA foreign_keys = ON")

        self._create_tables()

    def _create_tables(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,

                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,

                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,

                start_time TEXT,
                end_time TEXT,

                config_json TEXT NOT NULL
            );


            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,
                article_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- agent features

                direction REAL NOT NULL,

                positive_probability REAL,
                neutral_probability REAL,
                negative_probability REAL,

                materiality REAL NOT NULL,
                novelty REAL NOT NULL,
                certainty REAL NOT NULL,

                company_relevance REAL NOT NULL,
                market_relevance REAL NOT NULL,
                already_known REAL NOT NULL,

                event_type TEXT,
                effect_type TEXT,
                time_horizon TEXT,

                -- Market features

                return_since_news REAL,
                relative_volume REAL,
                return_1m REAL,
                return_5m REAL,
                realized_volatility REAL,
                spy_return_5m REAL,

                -- Score

                score REAL NOT NULL,

                -- Future outcomes

                future_return_1m REAL,
                future_return_5m REAL,
                future_return_15m REAL,
                future_return_30m REAL,
                future_return_60m REAL,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id),

                UNIQUE (
                    session_id,
                    article_id,
                    ticker
                )
            );


            CREATE INDEX IF NOT EXISTS
                idx_results_session
            ON results(session_id);


            CREATE INDEX IF NOT EXISTS
                idx_results_ticker
            ON results(ticker);


            CREATE INDEX IF NOT EXISTS
                idx_results_timestamp
            ON results(timestamp);
            """
        )

        self.connection.commit()

    def save_session(
            self,
            session: Session,
    ):
        config = {
            "name": session.config.name,
            "type": session.config.type.value,

            "start": self._datetime(
                session.config.start
            ),

            "end": self._datetime(
                session.config.end
            ),

            "weights": session.config.weights,
        }

        self.connection.execute(
            """
            INSERT INTO sessions (
                id,
                name,
                type,
                status,
                created_at,
                started_at,
                ended_at,
                start_time,
                end_time,
                config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                status = excluded.status,
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                start_time = excluded.start_time,
                end_time = excluded.end_time,
                config_json = excluded.config_json
            """,
            (
                session.id,
                session.config.name,
                session.config.type.value,
                session.status.value,

                self._datetime(
                    session.created_at
                ),

                self._datetime(
                    session.started_at
                ),

                self._datetime(
                    session.ended_at
                ),

                self._datetime(
                    session.config.start
                ),

                self._datetime(
                    session.config.end
                ),

                json.dumps(config),
            ),
        )

        self.connection.commit()

    def save_result(
            self,
            session_id: str,
            signal: NewsSignal,
            context: MarketContext,
            score: float,
    ):
        if signal.article_id != context.article_id:
            raise ValueError(
                "Signal and MarketContext belong "
                "to different articles"
            )

        if signal.ticker != context.ticker:
            raise ValueError(
                "Signal and MarketContext belong "
                "to different tickers"
            )

        self.connection.execute(
            """
            INSERT INTO results (
                session_id,
                article_id,
                ticker,
                timestamp,

                direction,

                positive_probability,
                neutral_probability,
                negative_probability,

                materiality,
                novelty,
                certainty,

                company_relevance,
                market_relevance,
                already_known,

                event_type,
                effect_type,
                time_horizon,

                return_since_news,
                relative_volume,
                return_1m,
                return_5m,
                realized_volatility,
                spy_return_5m,

                score
            )
            VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?
            )

            ON CONFLICT (
                session_id,
                article_id,
                ticker
            )
            DO UPDATE SET

                timestamp = excluded.timestamp,

                direction = excluded.direction,

                positive_probability =
                    excluded.positive_probability,

                neutral_probability =
                    excluded.neutral_probability,

                negative_probability =
                    excluded.negative_probability,

                materiality = excluded.materiality,
                novelty = excluded.novelty,
                certainty = excluded.certainty,

                company_relevance =
                    excluded.company_relevance,

                market_relevance =
                    excluded.market_relevance,

                already_known =
                    excluded.already_known,

                event_type = excluded.event_type,
                effect_type = excluded.effect_type,
                time_horizon = excluded.time_horizon,

                return_since_news =
                    excluded.return_since_news,

                relative_volume =
                    excluded.relative_volume,

                return_1m =
                    excluded.return_1m,

                return_5m =
                    excluded.return_5m,

                realized_volatility =
                    excluded.realized_volatility,

                spy_return_5m =
                    excluded.spy_return_5m,

                score = excluded.score
            """,
            (
                session_id,
                signal.article_id,
                signal.ticker,
                self._datetime(context.captured_at),

                signal.direction,

                signal.positive_probability,
                signal.neutral_probability,
                signal.negative_probability,

                signal.materiality,
                signal.novelty,
                signal.certainty,

                signal.company_relevance,
                signal.market_relevance,
                signal.already_known,

                signal.event_type,
                signal.effect_type,
                signal.time_horizon,

                self._finite(context.return_since_news),
                self._finite(context.relative_volume),
                self._finite(context.return_1m),
                self._finite(context.return_5m),
                self._finite(context.realized_volatility),
                self._finite(context.spy_return_5m),

                score,
            ),
        )

        self.connection.commit()

    def update_outcomes(
            self,
            session_id: str,
            article_id: int,
            ticker: str,
            *,
            return_1m: float | None = None,
            return_5m: float | None = None,
            return_15m: float | None = None,
            return_30m: float | None = None,
            return_60m: float | None = None,
    ):

        self.connection.execute(
            """
            UPDATE results
            SET
                future_return_1m = COALESCE(
                    ?, future_return_1m
                ),

                future_return_5m = COALESCE(
                    ?, future_return_5m
                ),

                future_return_15m = COALESCE(
                    ?, future_return_15m
                ),

                future_return_30m = COALESCE(
                    ?, future_return_30m
                ),

                future_return_60m = COALESCE(
                    ?, future_return_60m
                )

            WHERE
                session_id = ?
                AND article_id = ?
                AND ticker = ?
            """,
            (
                self._finite(return_1m),
                self._finite(return_5m),
                self._finite(return_15m),
                self._finite(return_30m),
                self._finite(return_60m),

                session_id,
                article_id,
                ticker,
            ),
        )

        self.connection.commit()

    def get_results(
            self,
            session_id: str,
    ) -> list[sqlite3.Row]:

        cursor = self.connection.execute(
            """
            SELECT *
            FROM results
            WHERE session_id = ?
            ORDER BY timestamp
            """,
            (session_id,),
        )

        return cursor.fetchall()

    def get_training_data(self) -> list[sqlite3.Row]:

        cursor = self.connection.execute(
            """
            SELECT *
            FROM results

            WHERE
                future_return_1m IS NOT NULL
                OR future_return_5m IS NOT NULL
                OR future_return_15m IS NOT NULL
                OR future_return_30m IS NOT NULL
                OR future_return_60m IS NOT NULL

            ORDER BY timestamp
            """
        )

        return cursor.fetchall()

    @staticmethod
    def _datetime(
            value: datetime | None,
    ) -> str | None:

        if value is None:
            return None

        return value.isoformat()

    @staticmethod
    def _finite(
            value: float | None,
    ) -> float | None:

        if value is None:
            return None

        # Converts nan/inf into SQL NULL.
        import math

        if not math.isfinite(value):
            return None

        return float(value)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
    ):
        self.close()
