"""Ticker-specific news classification adapters and orchestration."""


class ClassificationResponseError(ValueError):
    """The model returned data that cannot be converted to a classification."""
