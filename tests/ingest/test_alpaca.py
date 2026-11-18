from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from json import loads
from pathlib import Path
from typing import Final

import httpx
import pytest
from anyio import Path as AsyncPath
from anyio import fail_after
from websockets.asyncio.server import ServerConnection, serve

from optionsone.core.clock import SimClock
from optionsone.core.pit import PointInTimeNews
from optionsone.core.ports import HistoricalBars, HistoricalNews
from optionsone.ingest.alpaca_bars import AlpacaHistoricalBars
from optionsone.ingest.alpaca_news import AlpacaHistoricalNews, AlpacaNewsStream
from optionsone.ingest.retry import RetryPolicy, TokenBucket
from optionsone.ingest.store import ParquetStore

_FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "ingest"
_START: Final = datetime(2025, 7, 3, 13, 30, tzinfo=UTC)
_END: Final = datetime(2025, 7, 3, 17, tzinfo=UTC)


@pytest.fixture(autouse=True)
def alpaca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "fixture-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fixture-secret")


async def test_news_revisions_use_availability_time_and_survive_storage(tmp_path: Path) -> None:
    def serve_news(request: httpx.Request) -> httpx.Response:
        assert str(request.url.copy_with(query=None)) == "https://data.alpaca.markets/v1beta1/news"
        assert request.headers["APCA-API-KEY-ID"] == "fixture-key"
        assert request.headers["APCA-API-SECRET-KEY"] == "fixture-secret"
        assert request.url.params["start"] == _START.isoformat()
        assert request.url.params["end"] == _END.isoformat()
        assert request.url.params["sort"] == "asc"
        assert request.url.params["include_content"] == "true"
        token = request.url.params.get("page_token")
        fixture = {None: "news-1.json", "news-next": "news-2.json"}[token]
        return httpx.Response(200, content=(_FIXTURES / fixture).read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve_news)) as client:
        source: HistoricalNews = AlpacaHistoricalNews(client=client)
        revisions = await source.fetch(_START, _END)

    original_known = datetime(2025, 7, 3, 15, 0, 30, tzinfo=UTC)
    revision_known = datetime(2025, 7, 4, 2, 0, 30, tzinfo=UTC)
    assert [(a.id, a.knowledge_time) for a in revisions] == [
        (314159, original_known),
        (314159, revision_known),
    ]
    assert revisions[1].updated_at == datetime(2025, 7, 4, 2, tzinfo=UTC)
    assert revisions[1].content == "<p>Corrected results.</p>"
    assert revisions[1].symbols == ("AAPL",)

    store = ParquetStore(tmp_path)
    assert store.write_articles([*revisions, revisions[0]]) == 2
    original_files = {path async for path in AsyncPath(tmp_path).glob("news/date=*/*.parquet")}
    assert {path.parent.name for path in original_files} == {"date=2025-07-03", "date=2025-07-04"}
    assert store.write_articles(revisions) == 0
    assert {path async for path in AsyncPath(tmp_path).glob("news/date=*/*.parquet")} == (
        original_files
    )
    assert store.load_articles(original_known, revision_known) == [revisions[0]]
    all_revisions = store.load_articles(_START, revision_known + timedelta(seconds=1))
    assert all_revisions == revisions
    clock = SimClock(original_known)
    pit = PointInTimeNews(all_revisions, clock)
    assert pit.visible() == [revisions[0]]
    clock.advance_to(revision_known)
    assert pit.visible() == [revisions[1]]


async def test_bars_paginate_multi_symbol_iex_request_and_round_trip(tmp_path: Path) -> None:
    def serve_bars(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url.copy_with(query=None)) == "https://data.alpaca.markets/v2/stocks/bars"
        )
        assert request.url.params["symbols"] == "AAPL,MSFT"
        assert request.url.params["start"] == _START.isoformat()
        assert request.url.params["end"] == _END.isoformat()
        assert request.url.params["timeframe"] == "1Min"
        assert request.url.params["feed"] == "iex"
        token = request.url.params.get("page_token")
        fixture = {None: "bars-1.json", "bars-next": "bars-2.json"}[token]
        return httpx.Response(200, content=(_FIXTURES / fixture).read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve_bars)) as client:
        source: HistoricalBars = AlpacaHistoricalBars(client=client)
        bars = await source.fetch(["AAPL", "MSFT"], _START, _END)

    assert [(bar.symbol, bar.start, bar.close) for bar in bars] == [
        ("AAPL", _START, 201),
        ("AAPL", _START + timedelta(minutes=2), 202),
        ("MSFT", _START, 481),
    ]
    assert bars[0].vwap == 200.75
    assert bars[2].vwap is None
    store = ParquetStore(tmp_path)
    assert store.write_bars([*bars, bars[1]]) == 3
    assert store.write_bars(bars) == 0
    parts = [
        path
        async for path in AsyncPath(tmp_path).glob("bars/symbol=AAPL/date=2025-07-03/*.parquet")
    ]
    assert len(parts) == 1
    assert store.load_bars(["AAPL", "MSFT"], _START, _END) == bars
    assert store.load_bars(["AAPL"], _START, bars[1].start) == [bars[0]]


async def test_stream_reconnects_and_stamps_the_entire_frame_at_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    attempts = 0
    delays: list[float] = []
    receipt = datetime(2025, 7, 3, 15, 0, 10, tzinfo=UTC)
    clock = SimClock(receipt)

    async def wait(seconds: float) -> None:
        delays.append(seconds)

    async def handler(socket: ServerConnection) -> None:
        nonlocal attempts
        attempts += 1
        await socket.send('[{"T":"success","msg":"connected"}]')
        assert loads(await socket.recv()) == {
            "action": "auth",
            "key": "fixture-key",
            "secret": "fixture-secret",
        }
        await socket.send('[{"T":"success","msg":"authenticated"}]')
        assert loads(await socket.recv()) == {"action": "subscribe", "news": ["*"]}
        if attempts == 1:
            return
        await socket.send((_FIXTURES / "news-stream.json").read_text())
        await socket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        source = AlpacaNewsStream(clock=clock, wait=wait, url=f"ws://127.0.0.1:{port}/v1beta1/news")
        async with aclosing(source.stream()) as articles:
            with fail_after(5):
                first = await anext(articles)
                clock.advance_to(receipt + timedelta(seconds=5))
                second = await anext(articles)

    assert attempts == 2
    assert delays == [1.0]
    assert [(first.id, first.knowledge_time), (second.id, second.knowledge_time)] == [
        (8142, receipt),
        (8143, receipt),
    ]


@pytest.mark.parametrize(
    ("status", "header"),
    [(429, "7"), (503, "Thu, 03 Jul 2025 13:30:07 GMT")],
)
async def test_historical_news_obeys_retry_after(status: int, header: str) -> None:
    calls = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, headers={"Retry-After": header})
        return httpx.Response(200, content=(_FIXTURES / "news-2.json").read_bytes())

    async def wait(seconds: float) -> None:
        delays.append(seconds)

    retry = RetryPolicy(clock=SimClock(_START), wait=wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await AlpacaHistoricalNews(client=client, retry=retry).fetch(_START, _END)
    assert [article.id for article in result] == [314159]
    assert calls == 2
    assert delays == [7.0]


async def test_failed_attempts_spend_rate_budget_and_stop_at_the_limit() -> None:
    time = 0.0
    request_times: list[float] = []
    delay_times: list[float] = []

    def timer() -> float:
        return time

    async def advance(seconds: float) -> None:
        nonlocal time
        time += seconds

    async def retry_wait(seconds: float) -> None:
        delay_times.append(seconds)
        await advance(seconds)

    def unavailable(_: httpx.Request) -> httpx.Response:
        request_times.append(time)
        return httpx.Response(503)

    limiter = TokenBucket(requests_per_minute=1, timer=timer, wait=advance)
    retry = RetryPolicy(attempts=3, limiter=limiter, wait=retry_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        source = AlpacaHistoricalNews(client=client, retry=retry)
        with pytest.raises(httpx.HTTPStatusError) as failed:
            await source.fetch(_START, _END)

    assert failed.value.response.status_code == 503
    assert request_times == [0.0, 60.0, 120.0]
    assert delay_times == [1.0, 2.0]
