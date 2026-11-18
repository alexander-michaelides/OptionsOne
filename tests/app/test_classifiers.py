from pathlib import Path

import pytest

from optionsone.app.classifiers import ClassifierConfigurationError, make_classifier
from optionsone.classify.laya_adapter import LayaClassifier


def test_hosted_classifier_names_missing_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("OPTIONSONE_HOSTED_BASE_URL", "OPTIONSONE_HOSTED_MODEL", "OPTIONSONE_HOSTED_API_KEY")
    for name in names:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ClassifierConfigurationError) as caught:
        make_classifier("hosted")

    assert all(name in str(caught.value) for name in names)


def test_hosted_classifier_reads_settings_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONSONE_HOSTED_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPTIONSONE_HOSTED_MODEL", "test-model")
    monkeypatch.setenv("OPTIONSONE_HOSTED_API_KEY", "test-key")

    assert make_classifier("hosted").model_id == "test-model"


def test_laya_classifier_names_missing_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ClassifierConfigurationError, match="HF_TOKEN"):
        make_classifier("laya")


def test_invalid_hosted_settings_never_echo_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONSONE_HOSTED_BASE_URL", "not a url")
    monkeypatch.setenv("OPTIONSONE_HOSTED_MODEL", "m")
    monkeypatch.setenv("OPTIONSONE_HOSTED_API_KEY", "sk-SECRETVALUE")
    with pytest.raises(ClassifierConfigurationError) as info:
        make_classifier("hosted")
    assert "OPTIONSONE_HOSTED_BASE_URL" in str(info.value)
    assert "SECRETVALUE" not in str(info.value)
    assert info.value.__cause__ is None


def test_laya_token_is_read_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_dotenv\n")
    seen: list[str] = []

    def fake_from_laya(*, token: str) -> LayaClassifier:
        seen.append(token)
        raise RuntimeError("stop before loading a model")

    monkeypatch.setattr(LayaClassifier, "from_laya", staticmethod(fake_from_laya))
    with pytest.raises(RuntimeError, match="stop before"):
        make_classifier("laya")
    assert seen == ["hf_from_dotenv"]
