from collections.abc import Mapping
from datetime import timedelta
from threading import get_ident

import pytest

from optionsone.classify.laya_adapter import LayaClassifier
from optionsone.core.types import Article, EffectType


async def test_laya_predicts_off_thread_with_ticker_and_maps_legacy_answers(
    article: Article,
) -> None:
    calls: list[tuple[int, str, int]] = []

    class FakeRouter:
        def predict(self, state: dict[str, str], questions: Mapping[str, object], /) -> object:
            calls.append((get_ident(), state["ticker"], len(questions)))
            return {
                "answers": {
                    "event_type": {"choice": "legal"},
                    "direction": {"probabilities": {"positive": 0, "neutral": 1, "negative": 3}},
                    "materiality": {"score": 2},
                    "novelty": {"score": 3},
                    "certainty": {"score": 4},
                    "company_relevance": {"noul": 0.8},
                    "market_relevance": {"noul": 0.7},
                    "already_known": {"noul": 0.1},
                    "time_horizon": {"choice": "long"},
                    "effect_type": {"choice": "risk"},
                }
            }

    latency = timedelta(milliseconds=25)
    classifier = LayaClassifier(FakeRouter(), latency=latency)

    result = await classifier.classify(article, "BBB")

    [(worker, ticker, question_count)] = calls
    assert worker != get_ident()
    assert (ticker, question_count) == ("BBB", 10)
    assert result.ticker == "BBB"
    assert result.effect_type is EffectType.RISK
    assert result.direction == pytest.approx(-0.75)
    assert (result.materiality, result.novelty, result.certainty) == (0.5, 0.75, 1.0)
    assert result.knowledge_time == article.knowledge_time + latency


async def test_laya_expected_value_scores_map_continuously(article: Article) -> None:
    # Real laya 0.4.37 returns scores as expected values on the 0-4 scale, not whole levels.
    class RealShapeRouter:
        def predict(self, state: dict[str, str], questions: Mapping[str, object], /) -> object:
            return {
                "answers": {
                    "event_type": {"choice": "earnings"},
                    "direction": {
                        "probabilities": {"positive": 0.62, "neutral": 0.3, "negative": 0.08}
                    },
                    "materiality": {"score": 1.5932},
                    "novelty": {"score": 2.3575},
                    "certainty": {"score": 3.4744},
                    "company_relevance": {"noul": 0.91},
                    "market_relevance": {"noul": 0.84},
                    "already_known": {"noul": 0.12},
                    "time_horizon": {"choice": "short"},
                    "effect_type": {"choice": "revenue"},
                }
            }

    result = await LayaClassifier(RealShapeRouter(), latency=timedelta(0)).classify(article, "AAA")
    assert (result.materiality, result.novelty, result.certainty) == pytest.approx(
        (1.5932 / 4, 2.3575 / 4, 3.4744 / 4)
    )
    assert result.direction == pytest.approx(0.54)
