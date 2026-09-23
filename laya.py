from news import NewsArticle, NewsSignal

def analyze_article(article: NewsArticle) -> list[NewsSignal]:
    state = { # laya input state
        "headline": article.headline,
        "summary": article.summary,
        "content": article.content,
        "symbols": article.symbols,
    }