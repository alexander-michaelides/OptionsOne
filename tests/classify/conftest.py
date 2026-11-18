from datetime import UTC, datetime, timedelta

import pytest

from optionsone.core.types import Article


@pytest.fixture
def article() -> Article:
    published = datetime(2026, 4, 1, 14, 31, tzinfo=UTC)
    return Article(
        id=101,
        headline="Cloud provider beats expectations after sector report",
        summary="The report covered multiple companies.",
        content="The cloud provider reported results above consensus.",
        symbols=("AAA", "BBB", "CCC"),
        published_at=published,
        updated_at=published,
        knowledge_time=published + timedelta(seconds=4),
    )
