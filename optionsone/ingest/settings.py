from typing import Self

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingAlpacaCredentials(ValueError):
    """Raised without the offending values so secrets never reach logs or tracebacks."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__(f"invalid or missing Alpaca settings: {', '.join(names)}")


class LiveCredentialsForbidden(ValueError):
    def __init__(self) -> None:
        super().__init__("refusing an Alpaca live-account key (AK...); use a paper key (PK...)")


class AlpacaSettings(BaseSettings):
    """Load credentials from the environment or ./.env when an adapter is constructed."""

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_", env_file=".env", extra="ignore", frozen=True
    )

    api_key: SecretStr
    secret_key: SecretStr

    @classmethod
    def from_env(cls) -> Self:
        """Resolve settings sources; errors name the variables but never echo their values."""
        try:
            settings = cls.model_validate({})
        except ValidationError as exc:
            names = tuple(f"ALPACA_{str(e['loc'][0]).upper()}" for e in exc.errors())
            raise MissingAlpacaCredentials(names) from None
        if settings.api_key.get_secret_value().upper().startswith("AK"):
            raise LiveCredentialsForbidden()
        return settings

    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": self.secret_key.get_secret_value(),
        }
