"""Versioned questionnaire and per-ticker model input."""

from typing import Final

from optionsone.core.types import Article

PROMPT_VERSION: Final = "v1"

QUESTIONS: Final = {
    "event_type": {
        "type": "choice",
        "instructions": (
            "What is the primary financially relevant event described in this article?"
        ),
        "criteria": {
            "earnings": "Reported earnings or financial results",
            "guidance": "Forward-looking revenue, earnings, or other guidance",
            "analyst": "Analyst rating, recommendation, or price-target change",
            "contract": "Customer, supplier, or commercial contract",
            "capex": "Capital expenditure or infrastructure investment",
            "product": "Product launch, update, or development",
            "ma": "Merger, acquisition, divestiture, or takeover",
            "partnership": "Strategic partnership or collaboration",
            "legal": "Lawsuit or other legal development",
            "regulatory": "Government or regulatory action",
            "management": "Executive or management change",
            "financing": "Debt, equity issuance, buyback, or other financing",
            "operations": "Production, supply, manufacturing, or operational development",
            "macro": "Macroeconomic or industry-wide development",
            "other": "Other financially relevant event",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": (
            "What is the likely economic direction of this event for the specified company?"
        ),
        "criteria": {
            "negative": "Economically negative for the company",
            "neutral": "Little or no meaningful economic effect",
            "positive": "Economically positive for the company",
        },
    },
    "materiality": {
        "type": "score",
        "instructions": (
            "How economically significant could this event be for the specified company?"
        ),
        "criteria": ["negligible", "minor", "moderate", "major", "transformative"],
    },
    "novelty": {
        "type": "score",
        "instructions": (
            "How much genuinely new information does this article "
            "introduce relative to information described as already public?"
        ),
        "criteria": [
            "entirely repetitive",
            "mostly previously known",
            "partially new",
            "mostly new",
            "entirely new development",
        ],
    },
    "certainty": {
        "type": "score",
        "instructions": ("How certain is it that the event described has occurred or will occur?"),
        "criteria": [
            "rumor or speculation",
            "unconfirmed",
            "probable",
            "confirmed",
            "completed or directly reported fact",
        ],
    },
    "company_relevance": {
        "type": "noul",
        "instructions": (
            "Does this event have a meaningful economic connection to the specified company?"
        ),
    },
    "market_relevance": {
        "type": "noul",
        "instructions": (
            "Does this article contain information that investors "
            "could reasonably consider relevant when valuing the "
            "specified company?"
        ),
    },
    "already_known": {
        "type": "noul",
        "instructions": (
            "Does this article primarily discuss or repeat information "
            "that it describes as having already been publicly announced?"
        ),
    },
    "time_horizon": {
        "type": "choice",
        "instructions": (
            "Over what time horizon would the economic consequences "
            "of this event primarily affect the specified company?"
        ),
        "criteria": {
            "immediate": "Hours to days",
            "short": "Days to weeks",
            "medium": "Several weeks to months",
            "long": "Many months to years",
        },
    },
    "effect_type": {
        "type": "choice",
        "instructions": (
            "What is the primary mechanism through which this event "
            "could economically affect the specified company?"
        ),
        "criteria": {
            "revenue": "Changes expected revenue or demand",
            "cost": "Changes costs or margins",
            "capital": "Changes financing, cash, debt, or capital structure",
            "operations": "Changes production or operating capability",
            "competition": "Changes competitive position or market share",
            "risk": "Changes legal, regulatory, or business risk",
            "expectations": (
                "Primarily changes investor expectations without an immediate operational change"
            ),
            "none": "No clear economic mechanism",
        },
    },
}


def build_state(article: Article, ticker: str, company_name: str | None = None) -> dict[str, str]:
    """Build the article data sent to the model for one target company."""
    state = {
        "ticker": ticker,
        "headline": article.headline,
        "summary": article.summary,
        "content": article.content,
    }
    if company_name is not None:
        state["company_name"] = company_name
    return state
