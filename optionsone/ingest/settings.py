from typing import Self

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlpacaSettings(BaseSettings):
    """Load credentials from the environment when an adapter is constructed."""

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_", env_file=".env", extra="ignore", frozen=True
    )

    api_key: SecretStr
    secret_key: SecretStr

    @classmethod
    def from_env(cls) -> Self:
        """Run settings source resolution without explicit credential overrides."""
        return cls.model_validate({})

    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": self.secret_key.get_secret_value(),
        }
