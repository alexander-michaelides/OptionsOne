from pathlib import Path

import pytest

from optionsone.ingest.settings import (
    AlpacaSettings,
    LiveCredentialsForbidden,
    MissingAlpacaCredentials,
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_partial_credentials_error_never_echoes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "PKSECRETVALUE123")
    with pytest.raises(MissingAlpacaCredentials) as info:
        AlpacaSettings.from_env()
    assert "ALPACA_SECRET_KEY" in str(info.value)
    assert "PKSECRETVALUE123" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_live_key_refused_for_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "AKLIVE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    with pytest.raises(LiveCredentialsForbidden):
        AlpacaSettings.from_env()
