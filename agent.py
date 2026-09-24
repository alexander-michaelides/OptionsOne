import laya

from news import NewsArticle, NewsSignal
from laya import Router

QUESTIONS = {
    "event_type": {
        "type": "choice",
        "instructions": (
            "What is the primary financially relevant event "
            "described in this article?"
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
            "other": "Other financially relevant event"
        }
    },

    "direction": {
        "type": "choice",
        "instructions": (
            "What is the likely economic direction of this event "
            "for the specified company?"
        ),
        "criteria": {
            "negative": "Economically negative for the company",
            "neutral": "Little or no meaningful economic effect",
            "positive": "Economically positive for the company"
        }
    },

    "materiality": {
        "type": "score",
        "instructions": (
            "How economically significant could this event be "
            "for the specified company?"
        ),
        "criteria": [
            "negligible",
            "minor",
            "moderate",
            "major",
            "transformative"
        ]
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
            "entirely new development"
        ]
    },

    "certainty": {
        "type": "score",
        "instructions": (
            "How certain is it that the event described has occurred "
            "or will occur?"
        ),
        "criteria": [
            "rumor or speculation",
            "unconfirmed",
            "probable",
            "confirmed",
            "completed or directly reported fact"
        ]
    },

    "company_relevance": {
        "type": "noul",
        "instructions": (
            "Does this event have a meaningful economic connection "
            "to the specified company?"
        )
    },

    "market_relevance": {
        "type": "noul",
        "instructions": (
            "Does this article contain information that investors "
            "could reasonably consider relevant when valuing the "
            "specified company?"
        )
    },

    "already_known": {
        "type": "noul",
        "instructions": (
            "Does this article primarily discuss or repeat information "
            "that it describes as having already been publicly announced?"
        )
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
            "long": "Many months to years"
        }
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
                "Primarily changes investor expectations without "
                "an immediate operational change"
            ),
            "none": "No clear economic mechanism"
        }
    }
}

def analyze_article(agent: laya.Router, article: NewsArticle) -> list[NewsSignal]:
    tickers = []
    for ticker in article.symbols:
        state = { # laya input state
            "headline": article.headline,
            "summary": article.summary,
            "content": article.content,
        }
            
        answers = agent.predict(state, QUESTIONS)["answers"]
        
        # directions
        direction_probs = answers["direction"]["probabilities"]
        
        positive = direction_probs["positive"]
        negative = direction_probs["negative"]
        neutral = direction_probs["neutral"]
        
        direction = positive - negative # converts to -1 to 1 scale
        
        materiality = answers["materiality"]["score"] / 4
        novelty = answers["novelty"]["score"] / 4
        certainty = answers["certainty"]["score"] / 4
        
        # nouls
        company_relevance = answers["company_relevance"]["noul"]
        market_relevance = answers["market_relevance"]["noul"]
        already_known = answers["already_known"]["noul"]
        
        # choices
        event_type = answers["event_type"]["choice"]
        time_horizon = answers["time_horizon"]["choice"]
        effect_type = answers["effect_type"]["choice"]
        
        tickers.append(NewsSignal(
            article_id=article.article_id,
            ticker=ticker,
            
            event_type=event_type,
            effect_type=effect_type,
            time_horizon=time_horizon,
            
            direction=direction,
            positive_probability=positive,
            neutral_probability=neutral,
            negative_probability=negative,
            
            materiality=materiality,
            novelty=novelty,
            certainty=certainty,
            
            company_relevance=company_relevance,
            market_relevance=market_relevance,
            already_known=already_known
        ))

    return tickers