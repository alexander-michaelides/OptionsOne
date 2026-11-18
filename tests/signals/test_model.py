from datetime import UTC, datetime

import pytest

from optionsone.core.types import FeatureVector, Side
from optionsone.signals.model import LinearSignalModel, RidgeSpec
from optionsone.signals.rules import passes_pre_trade

NOW = datetime(2025, 1, 2, 15, tzinfo=UTC)


def feature(direction: float) -> FeatureVector:
    return FeatureVector(
        article_id=1,
        ticker="ABC",
        as_of=NOW,
        values={"direction": direction, "company_relevance": 0.7, "novelty": 0.8},
    )


def test_model_json_roundtrip_preserves_predictions_and_hash() -> None:
    model = LinearSignalModel.fit(
        [feature(-1), feature(0), feature(1)],
        [-0.01, 0.0, 0.01],
        RidgeSpec(feature_names=("direction",), threshold=0.5),
    )
    restored = LinearSignalModel.from_json(model.to_json())
    assert restored.model_id == model.model_id
    for direction, side in [(-1.0, Side.SHORT), (1.0, Side.LONG)]:
        signal = model.predict(feature(direction))
        assert signal is not None
        assert signal.side == side
        assert signal.strength == 1
        assert restored.predict(feature(direction)) == signal
    assert restored.predict(feature(0.1)) is None


def test_training_statistics_exclude_holdout_and_retain_neutral_target() -> None:
    model = LinearSignalModel.fit(
        [feature(-1), feature(0), feature(1)],
        [0.0, 0.06, 0.03],
        RidgeSpec(feature_names=("direction", "novelty"), alpha=1.0),
    )
    assert model.means == pytest.approx((0.0, 0.8))
    assert model.scales == pytest.approx(((2 / 3) ** 0.5, 1.0))
    assert model.score(feature(0)) == pytest.approx(0.03)
    assert model.score(feature(100)) == pytest.approx(0.03 + 100 * 0.01125)
    assert model.score(feature(0)) == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("company", "novelty", "is_open", "expected"),
    [
        (0.7, 0.8, True, True),
        (0.71, 0.8, True, False),
        (0.7, 0.81, True, False),
        (0.7, 0.8, False, False),
    ],
)
def test_pre_trade_requires_both_cutoffs_and_regular_session(
    company: float, novelty: float, is_open: bool, expected: bool
) -> None:
    assert (
        passes_pre_trade(
            feature(1),
            lambda timestamp: timestamp == NOW and is_open,
            min_company_relevance=company,
            min_novelty=novelty,
        )
        is expected
    )
