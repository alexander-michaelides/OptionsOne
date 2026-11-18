import asyncio
import argparse
import os
import warnings
from datetime import datetime, timezone

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings(
    "ignore",
    message="laya: this checkpoint ships invalid temperatures.*",
)

import dotenv
dotenv.load_dotenv()

from laya import Router

import agent
import market
import scoring
from session import Session, SessionConfig, SessionType
from storage import Storage


async def run_historical(resume_session_id: str | None = None):
    print("Initializing...")
    router = Router(preload=True, token=os.getenv("HF_TOKEN"))
    print("Router initialized.")

    storage = Storage()
    if resume_session_id:
        session = Session.resume(resume_session_id, storage)
        config = session.config
        print(f"Resuming session: {session.id}")
    else:
        config = SessionConfig(
            name="historical-test",
            type=SessionType.HISTORICAL,
            start=datetime(2026, 9, 20, 14, 0, tzinfo=timezone.utc),
            end=datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc),
        )
        session = Session(config, storage)
        storage.save_session(session)

    session.start()

    print(f"Session: {session.id}")
    print(f"Range:   {config.start} -> {config.end}")
    print()

    try:
        async for article in session.news():
            current_time = session.now()

            signals = agent.analyze_article(router, article)
            contexts = market.create_from_article(article, current_time)
            contexts_by_ticker = {
                context.ticker: context
                for context in contexts
            }

            for signal in signals:
                context = contexts_by_ticker.get(signal.ticker)
                if context is None:
                    continue

                quality = scoring.score(signal, context)

                storage.save_result(session.id, signal, context, quality)

                outcome = market.create_outcome(
                    article_id=article.article_id,
                    ticker=signal.ticker,
                    current_time=current_time,
                )

                storage.update_outcomes(
                    session_id=session.id,
                    article_id=article.article_id,
                    ticker=signal.ticker,

                    return_1m=outcome.return_1m,
                    return_5m=outcome.return_5m,
                    return_15m=outcome.return_15m,
                    return_30m=outcome.return_30m,
                    return_60m=outcome.return_60m,
                )

                future_return = outcome.return_15m

                if future_return is not None:
                    predicted_direction = (
                        "LONG"
                        if signal.direction > 0
                        else "SHORT"
                    )

                    trade_return = (
                        future_return
                        if signal.direction > 0
                        else -future_return
                    )

                    correct = trade_return > 0

                    print(
                        f"{signal.ticker:<6} | "
                        f"{predicted_direction:<5} | "
                        f"Score: {quality:.2f} | "
                        f"15m: {future_return:+.2%} | "
                        f"Trade: {trade_return:+.2%} | "
                        f"{'✓ CORRECT' if correct else '✗ WRONG'}"
                    )

            session.mark_processed(article)
            print(
                f"Progress: {session.progress():6.1%} | "
                f"Articles: {session.articles_processed} | "
                f"Through: {current_time.isoformat(timespec='seconds')}"
            )

        session.complete()
        print(
            f"Progress: 100.0% | Articles: {session.articles_processed} | Complete"
        )
    except Exception:
        session.fail()
        raise
    finally:
        evaluate_results(storage, session.id)
        storage.close()


def evaluate_results(storage, session_id: str):
    rows = storage.get_results(session_id)

    trades = []

    for row in rows:
        future_return = row["future_return_15m"]

        if future_return is None:
            continue

        direction = row["direction"]
        score = row["score"]

        # Ignore signals with effectively no predicted direction
        if direction == 0:
            continue

        # Convert stock return into return in the direction
        # the agent predicted.
        trade_return = (
            future_return
            if direction > 0
            else -future_return
        )

        trades.append({
            "score": score,
            "correct": trade_return > 0,
            "return": trade_return,
        })

    if not trades:
        print("No completed results.")
        return

    accuracy = (
        sum(t["correct"] for t in trades)
        / len(trades)
    )

    avg_return = (
        sum(t["return"] for t in trades)
        / len(trades)
    )

    print("\n" + "=" * 50)
    print("AGENT PERFORMANCE")
    print("=" * 50)

    print(f"Signals:        {len(trades)}")
    print(f"Direction acc:  {accuracy:.1%}")
    print(f"Avg return:     {avg_return:+.3%}")

    # Performance at different score thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:

        selected = [
            t
            for t in trades
            if t["score"] >= threshold
        ]

        if not selected:
            continue

        accuracy = (
            sum(t["correct"] for t in selected)
            / len(selected)
        )

        avg_return = (
            sum(t["return"] for t in selected)
            / len(selected)
        )

        print(
            f"Score >= {threshold:.1f}: "
            f"{len(selected):4} signals | "
            f"{accuracy:6.1%} correct | "
            f"{avg_return:+7.3%} avg"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect and score historical news market data."
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="continue an incomplete historical session by its printed ID",
    )
    args = parser.parse_args()
    asyncio.run(run_historical(args.resume))
