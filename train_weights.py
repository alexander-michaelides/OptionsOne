import sqlite3

import numpy as np
from scipy.optimize import minimize

from scoring import ScoreWeights


DATABASE = "data/trading.db"


# ============================================================
# Same transformations used by scoring.py
# ============================================================

def freshness(return_since_news: float) -> float:
    move = abs(return_since_news)

    if move < 0.0025:
        return 1.00
    if move < 0.005:
        return 0.90
    if move < 0.010:
        return 0.70
    if move < 0.020:
        return 0.40

    return 0.15


def volume_quality(relative_volume: float) -> float:
    if relative_volume >= 2.0:
        return 1.00
    if relative_volume >= 1.5:
        return 0.90
    if relative_volume >= 1.0:
        return 0.75
    if relative_volume >= 0.5:
        return 0.50

    return 0.25


def alignment(
    direction: float,
    return_5m: float | None,
) -> float:

    if return_5m is None:
        return 0.50

    if direction * return_5m > 0:
        return 1.00

    if abs(return_5m) < 0.001:
        return 0.70

    return 0.30


# ============================================================
# Load historical results
# ============================================================

connection = sqlite3.connect(DATABASE)
connection.row_factory = sqlite3.Row

rows = connection.execute(
    """
    SELECT *
    FROM results
    WHERE future_return_15m IS NOT NULL
    ORDER BY timestamp
    """
).fetchall()

connection.close()


print(f"Loaded {len(rows)} training examples.")


# ============================================================
# Build dataset
# ============================================================

X = []
y = []


for row in rows:

    features = [
        abs(row["direction"]),
        row["materiality"],
        row["novelty"],
        row["certainty"],
        row["company_relevance"],
        row["market_relevance"],

        # Higher should always mean better.
        1.0 - row["already_known"],

        freshness(
            row["return_since_news"]
        ),

        volume_quality(
            row["relative_volume"]
        ),

        alignment(
            row["direction"],
            row["return_5m"],
        ),
    ]

    X.append(features)

    # Return from the direction the agent predicted.
    if row["direction"] > 0:
        trade_return = row["future_return_15m"]

    elif row["direction"] < 0:
        trade_return = -row["future_return_15m"]

    else:
        # Neutral signal doesn't have a trade direction.
        continue

    y.append(trade_return)


X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)


# ============================================================
# Train/test split
#
# Data is already sorted chronologically.
# DO NOT randomly shuffle time-series data.
# ============================================================

split = int(len(X) * 0.80)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


print(f"Train: {len(X_train)}")
print(f"Test:  {len(X_test)}")


# ============================================================
# Optimization objective
# ============================================================

def objective(weights):

    scores = X_train @ weights

    correlation = np.corrcoef(
        scores,
        y_train,
    )[0, 1]

    if not np.isfinite(correlation):
        return 1.0

    # scipy minimizes.
    return -correlation


# ============================================================
# Initial weights
# ============================================================

initial_weights = np.array([
    0.20,  # direction
    0.15,  # materiality
    0.12,  # novelty
    0.08,  # certainty
    0.07,  # company relevance
    0.07,  # market relevance
    0.10,  # not already known
    0.08,  # freshness
    0.07,  # relative volume
    0.06,  # alignment
])


# ============================================================
# Optimize
# ============================================================

optimization_iterations = 0


def show_optimization_progress(weights):
    global optimization_iterations
    optimization_iterations += 1
    scores = X_train @ weights
    correlation = np.corrcoef(scores, y_train)[0, 1]
    correlation_text = f"{correlation:+.4f}" if np.isfinite(correlation) else "n/a"
    print(
        f"\rOptimizing weights: iteration {optimization_iterations} | "
        f"train correlation: {correlation_text}",
        end="",
        flush=True,
    )


result = minimize(
    objective,
    initial_weights,

    method="SLSQP",

    bounds=[
        (0.0, 1.0)
        for _ in initial_weights
    ],

    constraints={
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1.0,
    },
    callback=show_optimization_progress,
)
print(f"\rWeight optimization finished after {optimization_iterations} iterations.")


weights = result.x


# ============================================================
# Evaluate
# ============================================================

train_scores = X_train @ weights
test_scores = X_test @ weights


train_correlation = np.corrcoef(
    train_scores,
    y_train,
)[0, 1]

test_correlation = np.corrcoef(
    test_scores,
    y_test,
)[0, 1]


# ============================================================
# Print
# ============================================================

names = [
    "direction",
    "materiality",
    "novelty",
    "certainty",
    "company_relevance",
    "market_relevance",
    "already_known",
    "freshness",
    "relative_volume",
    "alignment",
]


print("\nTRAINED WEIGHTS")
print("=" * 40)

for name, weight in zip(
    names,
    weights,
):
    print(
        f"{name:<20} {weight:.4f}"
    )


print("\nPERFORMANCE")
print("=" * 40)

print(
    f"Train correlation: "
    f"{train_correlation:+.4f}"
)

print(
    f"Test correlation:  "
    f"{test_correlation:+.4f}"
)


print("\nScoreWeights(")

for name, weight in zip(
    names,
    weights,
):
    print(
        f"    {name}={weight:.4f},"
    )

print(")")
