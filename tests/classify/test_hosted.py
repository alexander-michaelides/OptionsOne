import json
from datetime import timedelta
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, HttpUrl, SecretStr
from respx import MockRouter

from optionsone.classify import ClassificationResponseError
from optionsone.classify.hosted import HostedClassifier, HostedSettings
from optionsone.core.clock import SimClock
from optionsone.core.types import Article, EffectType, EventType, TimeHorizon

_BASE_URL: Final = "https://classify.example/v1/"
_ANSWERS: Final = {
    "event_type": "earnings",
    "direction": {"positive": 6, "neutral": 3, "negative": 1},
    "materiality": 3,
    "novelty": 2,
    "certainty": 4,
    "company_relevance": 0.9,
    "market_relevance": 0.7,
    "already_known": 0.1,
    "time_horizon": "short",
    "effect_type": "revenue",
}


class _State(BaseModel):
    ticker: str
    company_name: str | None = None


class _Message(BaseModel):
    role: str
    content: str


class _Schema(BaseModel):
    strict: bool


class _Format(BaseModel):
    type: str
    json_schema: _Schema


class _Request(BaseModel):
    model: str
    messages: list[_Message]
    response_format: _Format


@pytest.fixture
def settings() -> HostedSettings:
    return HostedSettings(
        base_url=HttpUrl(_BASE_URL), model="offline-model", api_key=SecretStr("test-token")
    )


def _completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_same_article_sends_distinct_target_states(
    article: Article, settings: HostedSettings, respx_mock: MockRouter
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _completion(json.dumps(_ANSWERS))

    respx_mock.post(f"{_BASE_URL}chat/completions").mock(side_effect=respond)
    classifier = HostedClassifier(settings, latency=timedelta(0))

    await classifier.classify(article, "AAA", company_name="First Company")
    await classifier.classify(article, "BBB")

    assert len(requests) == 2
    assert all(request.headers["Authorization"] == "Bearer test-token" for request in requests)
    assert requests[0].content != requests[1].content
    targets: list[_State] = []
    for request in requests:
        payload = _Request.model_validate_json(request.content)
        assert payload.model == "offline-model"
        assert payload.response_format.type == "json_schema"
        assert payload.response_format.json_schema.strict
        targets.extend(
            _State.model_validate_json(message.content)
            for message in payload.messages
            if message.role == "user"
        )
    assert targets == [
        _State(ticker="AAA", company_name="First Company"),
        _State(ticker="BBB"),
    ]


@pytest.mark.parametrize("fixed_latency", [None, timedelta(milliseconds=17), timedelta(0)])
async def test_hosted_maps_answers_and_uses_configured_latency_source(
    article: Article,
    settings: HostedSettings,
    respx_mock: MockRouter,
    fixed_latency: timedelta | None,
) -> None:
    clock = SimClock(article.knowledge_time + timedelta(minutes=5))
    measured = timedelta(milliseconds=731)

    def respond(request: httpx.Request) -> httpx.Response:
        clock.advance_to(clock.now() + measured)
        return _completion(json.dumps(_ANSWERS))

    respx_mock.post(f"{_BASE_URL}chat/completions").mock(side_effect=respond)
    classifier = HostedClassifier(settings, clock=clock, latency=fixed_latency)

    result = await classifier.classify(article, "AAA")

    assert (result.article_id, result.ticker, result.model_id) == (
        article.id,
        "AAA",
        "offline-model",
    )
    assert (result.event_type, result.effect_type, result.time_horizon) == (
        EventType.EARNINGS,
        EffectType.REVENUE,
        TimeHorizon.SHORT,
    )
    assert (result.p_positive, result.p_neutral, result.p_negative) == pytest.approx(
        (0.6, 0.3, 0.1)
    )
    assert (result.materiality, result.novelty, result.certainty) == (0.75, 0.5, 1.0)
    assert (result.company_relevance, result.market_relevance, result.already_known) == (
        0.9,
        0.7,
        0.1,
    )
    expected = measured if fixed_latency is None else fixed_latency
    assert result.knowledge_time == article.knowledge_time + expected


@pytest.mark.parametrize(
    "content",
    [
        "broken JSON {",
        json.dumps({**_ANSWERS, "materiality": 5}),
        json.dumps({**_ANSWERS, "direction": {"positive": 0, "neutral": 0, "negative": 0}}),
    ],
)
async def test_hosted_rejects_invalid_model_answers_with_typed_error(
    article: Article, settings: HostedSettings, respx_mock: MockRouter, content: str
) -> None:
    respx_mock.post(f"{_BASE_URL}chat/completions").mock(return_value=_completion(content))

    with pytest.raises(ClassificationResponseError):
        await HostedClassifier(settings).classify(article, "AAA")
