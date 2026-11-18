"""Latched equity-based circuit breaker."""

from datetime import date
from hmac import compare_digest

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from optionsone.core.types import OrderIntent
from optionsone.risk.limits import reduces_position


class KillSwitchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1)
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1)
    reset_code: SecretStr = Field(min_length=1)


class KillSwitch:
    """Track session opening equity and the high-water mark until manual reset."""

    def __init__(self, config: KillSwitchConfig) -> None:
        self._config = config
        self._reason: str | None = None
        self._session: date | None = None
        self._opening_equity = 0.0
        self._peak = 0.0

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def flatten_requested(self) -> bool:
        return self._reason is not None

    def observe(self, session: date, equity: float) -> None:
        if self._session != session:
            self._session = session
            self._opening_equity = equity
        self._peak = max(self._peak, equity)
        if self._reason is None:
            if self._opening_equity > 0 and equity <= self._opening_equity * (
                1 - self._config.max_daily_loss_pct
            ):
                self._reason = "daily_loss"
            elif self._peak > 0 and equity <= self._peak * (1 - self._config.max_drawdown_pct):
                self._reason = "peak_drawdown"

    def allows(self, order: OrderIntent, current_qty: int) -> bool:
        return not self.flatten_requested or reduces_position(order, current_qty)

    def reset(self, code: str) -> bool:
        if not compare_digest(code.encode(), self._config.reset_code.get_secret_value().encode()):
            return False
        self._reason = None
        self._session = None
        self._peak = 0.0
        return True
