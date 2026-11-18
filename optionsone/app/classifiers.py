"""Construct the classifiers available to application workflows."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Literal, assert_never

from pydantic import ValidationError

from optionsone.classify.cache import CachedClassifier
from optionsone.classify.hosted import HostedClassifier, HostedSettings
from optionsone.classify.laya_adapter import LayaClassifier
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.ports import Classifier


@dataclass(frozen=True, slots=True)
class ClassifierConfigurationError(ValueError):
    missing: tuple[str, ...]

    def __str__(self) -> str:
        return f"Missing classifier environment variables: {', '.join(self.missing)}"


def make_classifier(
    kind: Literal["stub", "laya", "hosted"], cache_path: Path | None = None
) -> Classifier:
    """Resolve classifier settings at construction and optionally persist its results."""
    classifier: Classifier
    match kind:
        case "stub":
            classifier = DeterministicStubClassifier()
        case "laya":
            token = getenv("HF_TOKEN")
            if not token:
                raise ClassifierConfigurationError(("HF_TOKEN",))
            classifier = LayaClassifier.from_laya(token=token)
        case "hosted":
            try:
                settings = HostedSettings.model_validate({})
            except ValidationError as exc:
                missing = tuple(
                    f"OPTIONSONE_HOSTED_{str(issue['loc'][0]).upper()}"
                    for issue in exc.errors()
                    if issue["type"] == "missing"
                )
                if missing:
                    raise ClassifierConfigurationError(missing) from exc
                raise
            classifier = HostedClassifier(settings)
        case _:
            assert_never(kind)

    if cache_path is None:
        return classifier
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return CachedClassifier(classifier, cache_path)
