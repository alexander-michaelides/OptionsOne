"""Ridge predictions from immutable, content-addressed model parameters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from hashlib import sha256
from typing import Protocol, Self

import numpy as np
from pydantic import Field, FiniteFloat, PositiveFloat, model_validator

from optionsone.core.types import FeatureVector, Frozen, Side, Signal, Unit
from optionsone.features.pipeline import FEATURE_NAMES


class SignalModel(Protocol):
    def predict(self, fv: FeatureVector) -> Signal | None: ...


class RidgeSpec(Frozen):
    alpha: PositiveFloat = 1.0
    horizon: timedelta = Field(default=timedelta(minutes=15), gt=timedelta(0))
    threshold: Unit = 0.0
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be nonempty and unique")
        return self


DEFAULT_RIDGE_SPEC = RidgeSpec()


class LinearSignalModel(Frozen):
    spec: RidgeSpec
    means: tuple[FiniteFloat, ...]
    scales: tuple[PositiveFloat, ...]
    weights: tuple[FiniteFloat, ...]
    intercept: FiniteFloat
    strength_scale: PositiveFloat

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        if (
            not len(self.spec.feature_names)
            == len(self.means)
            == len(self.scales)
            == len(self.weights)
        ):
            raise ValueError("model parameters must match the feature schema")
        if not all(np.isfinite(value) for value in (*self.scales, self.strength_scale)):
            raise ValueError("scales must be finite")
        return self

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[FeatureVector],
        targets: Sequence[float],
        spec: RidgeSpec = DEFAULT_RIDGE_SPEC,
    ) -> Self:
        """Fit signed returns, retaining all supplied rows, including neutral events."""
        if not feature_rows or len(feature_rows) != len(targets):
            raise ValueError("fit requires nonempty aligned features and targets")
        x = np.array(
            [[row.values[name] for name in spec.feature_names] for row in feature_rows],
            dtype=np.float64,
        )
        y = np.array(targets, dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("fit requires finite features and targets")
        varies = np.ptp(x, axis=0) > 0
        means = np.where(varies, x.mean(axis=0), x[0])
        scales = np.where(varies, x.std(axis=0), 1.0)
        z = (x - means) / scales
        intercept = float(y.mean())
        weights = np.linalg.solve(z.T @ z + spec.alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
        strength_scale = float(np.quantile(np.abs(intercept + z @ weights), 0.9))
        return cls(
            spec=spec,
            means=tuple(map(float, means)),
            scales=tuple(map(float, scales)),
            weights=tuple(map(float, weights)),
            intercept=intercept,
            strength_scale=strength_scale or 1.0,
        )

    @property
    def model_id(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)

    def score(self, fv: FeatureVector) -> float:
        """Predict the signed target before strength calibration or thresholding."""
        return self.intercept + sum(
            (fv.values[name] - mean) / scale * weight
            for name, mean, scale, weight in zip(
                self.spec.feature_names, self.means, self.scales, self.weights, strict=True
            )
        )

    def predict(self, fv: FeatureVector) -> Signal | None:
        """Scale magnitude by the training 90th percentile and clip to one."""
        prediction = self.score(fv)
        if not np.isfinite(prediction) or prediction == 0:
            return None
        strength = min(1.0, abs(prediction) / self.strength_scale)
        if strength < self.spec.threshold:
            return None
        return Signal(
            article_id=fv.article_id,
            ticker=fv.ticker,
            as_of=fv.as_of,
            horizon=self.spec.horizon,
            side=Side.LONG if prediction > 0 else Side.SHORT,
            strength=strength,
            model_id=self.model_id,
        )
