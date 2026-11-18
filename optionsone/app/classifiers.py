"""Construct the classifiers available to application workflows."""

from pathlib import Path
from typing import Literal, assert_never

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from optionsone.classify.cache import CachedClassifier
from optionsone.classify.hosted import HostedClassifier, HostedSettings
from optionsone.classify.laya_adapter import LayaClassifier
from optionsone.classify.stub import DeterministicStubClassifier
from optionsone.core.ports import Classifier


class _HuggingFaceSettings(BaseSettings):
    """HF_TOKEN from the environment or ./.env, like the Alpaca credentials."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    hf_token: SecretStr = SecretStr("")


class ClassifierConfigurationError(ValueError):
    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(
            f"Missing or invalid classifier environment variables: {', '.join(missing)}"
        )


def make_classifier(
    kind: Literal["stub", "laya", "hosted"], cache_path: Path | None = None
) -> Classifier:
    """Resolve classifier settings at construction and optionally persist its results."""
    classifier: Classifier
    match kind:
        case "stub":
            classifier = DeterministicStubClassifier()
        case "laya":
            token = _HuggingFaceSettings().hf_token.get_secret_value()
            if not token:
                raise ClassifierConfigurationError(("HF_TOKEN",))
            classifier = LayaClassifier.from_laya(token=token)
        case "hosted":
            try:
                settings = HostedSettings.model_validate({})
            except ValidationError as exc:
                # Name the variables only; pydantic's error text echoes the configured values.
                invalid = tuple(
                    f"OPTIONSONE_HOSTED_{str(issue['loc'][0]).upper()}" for issue in exc.errors()
                )
                raise ClassifierConfigurationError(invalid) from None
            classifier = HostedClassifier(settings)
        case _:
            assert_never(kind)

    if cache_path is None:
        return classifier
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return CachedClassifier(classifier, cache_path)
