import pytest

from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.types import Article


@pytest.mark.parametrize(
    ("headline", "probability"),
    [
        ("Company beats expectations and raises guidance", "p_positive"),
        ("Company misses expectations and faces lawsuit", "p_negative"),
        ("Company publishes report", "p_neutral"),
    ],
)
async def test_stub_is_reproducible_and_reflects_headline_polarity(
    article: Article, headline: str, probability: str
) -> None:
    revision = article.model_copy(update={"headline": headline})
    classifier = DeterministicStubClassifier()

    result = await classifier.classify(revision, "AAA")
    replay = await DeterministicStubClassifier().classify(revision, "AAA")

    assert result == replay
    probabilities = {
        "p_positive": result.p_positive,
        "p_negative": result.p_negative,
        "p_neutral": result.p_neutral,
    }
    assert max(probabilities, key=probabilities.__getitem__) == probability
    assert result.knowledge_time == revision.knowledge_time
