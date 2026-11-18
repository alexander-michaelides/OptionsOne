# OptionsOne

News-driven intraday US-equity research and **paper-trading** system.

Alpaca news -> LLM event classification (per ticker) -> point-in-time features ->
signal model -> risk (sizing, limits, kill switch) -> execution (simulated or Alpaca paper).

## Layout

| Package | Responsibility |
|---|---|
| `optionsone/core` | Immutable domain types, `Clock` (only place that reads wall time), point-in-time views (`PointInTimeBars`, `PointInTimeNews`), ports (protocols) |
| `optionsone/ingest` | Alpaca news REST + WebSocket, IEX 1-minute bars, Parquet store (keeps every article revision), NYSE calendar, data validation |
| `optionsone/classify` | Versioned 10-question schema, hosted OpenAI-compatible adapter, local `laya` adapter, deterministic stub, SQLite cache, bounded batcher |
| `optionsone/features` | Single feature pipeline used by research and live; reads bars only through `PointInTimeBars` |
| `optionsone/signals` | Serializable ridge signal model (content-hashed `model_id`), pre-trade rules |
| `optionsone/research` | Forward / market-adjusted / triple-barrier labels, purged walk-forward CV, Rank IC + HAC t, deflated Sharpe, trial registry, promotion gate |
| `optionsone/risk` | Volatility-targeted sizing, limits, latched kill switch |
| `optionsone/execution` | Cost model, `SimBroker`, order manager (idempotent ids), `AlpacaPaperBroker` (paper endpoint only) |
| `optionsone/backtest` | Event-driven replay on `SimClock`, portfolio, report |
| `optionsone/app` | `optionsone` CLI |
| `legacy/` | The original scripts and `trading.db`, kept for reference only |

## Setup

```bash
uv sync
uv run pytest -q && uv run ruff check . && uv run pyright
uv run optionsone backtest --fixture     # offline end-to-end run, no credentials needed
```

Environment variables (never read at import time):

- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`: market data and news (free IEX plan is enough).
- `ALPACA_PAPER_API_KEY`, `ALPACA_PAPER_SECRET_KEY`: paper trading account.
- Hosted LLM settings for the OpenAI-compatible adapter: base URL, model, and API key (see `optionsone/classify/hosted.py`).

## Paper trading runbook

Credentials come from the environment or `./.env`: `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET_KEY`, or the
generic `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` when they are **paper** keys (`PK...`). Live keys (`AK...`) are
refused for both trading and market data, and configuration errors name variables without echoing their values.
Market data is the free IEX feed. Everything writes under `data/` (gitignored).

```bash
uv run optionsone paper-check        # account ACTIVE, market clock, positions, IEX bars reachable
uv run optionsone paper-smoke        # 1-share SPY limit at 50% of last price -> verify open -> cancel

# 1. Backfill a research window (news + 1-minute bars for the universe and SPY). Prints one line per
#    day and writes each day before fetching the next, so Ctrl-C or a crash loses at most one day; re-running
#    the same command skips completed days (data/store/_backfill_done.jsonl). Writes are deduplicated.
uv run optionsone backfill --start 2026-08-03 --end 2026-11-14 --symbols AAPL,NVDA,TSLA,AMD,MSFT

# 2. Research: classify (cached), PIT features, purged walk-forward CV, fit + save the model.
#    Use --classifier laya (needs HF_TOKEN) or hosted (OPTIONSONE_HOSTED_* settings); the stub has no edge.
uv run optionsone research --start 2026-08-03 --end 2026-11-14 --classifier laya
#    -> data/models/model.json and model.report.json (promotion_gate PASS/FAIL)

# 2b. Event-driven backtest on the stored data: fit on the first half, replay the second half through the
#     engine (next-open fills, costs, sizing, limits, pre-close flatten). Skipped signals are listed by reason.
uv run optionsone backtest --start 2026-08-03 --end 2026-11-14 --classifier laya --initial-cash 1000
#     size_below_one_share means the stock costs more than one position allows (10% of equity by default).

# 3. Paper trade during market hours (Ctrl-C stops gracefully; working orders are cancelled).
uv run optionsone paper --symbols AAPL,NVDA,TSLA,AMD,MSFT --classifier laya
#    --allow-unpromoted   trade a model that failed the gate (plumbing tests only)
#    --max-minutes N      bounded run;  --flatten-on-stop  close owned positions on exit
#    --max-position-pct / --max-open-positions  risk caps (defaults 10% / 5)
#    A kill-switch breach (3% daily loss / 15% drawdown) latches for the rest of the run; review the
#    journal, then restart to clear it.
```

What the paper runner does: for each new article (deduped by id and revision) mentioning a universe symbol, it
classifies the article for that ticker, fetches the last 45 minutes of completed IEX bars, runs the same `decide()` as
the backtest, then sizes the trade (volatility targeted, skipped below one share), checks limits and the latched kill
switch, and submits an idempotent market order. A 5-second ticker collects fills, submits horizon exits, and flattens
the runner's own positions 5 minutes before the NYSE close. Every decision, including each skip and its reason, is
appended to `data/journal/paper-<timestamp>.jsonl`.

The runner only manages positions and orders it created. Positions already in the account are journaled as foreign
and never traded, flattened or cancelled; symbols that hold one are skipped.

## Safety guarantees

- **No live trading exists.** `AlpacaPaperBroker` accepts only `https://paper-api.alpaca.markets` and raises
  `LiveTradingForbidden` for anything else. There is no live broker class.
- **Simulation is isolated.** `optionsone/backtest` must not import the Alpaca broker; a test checks this by
  parsing the imports.
- **No lookahead.** Bars become visible at their *end* time. Articles become visible at
  `knowledge_time = max(published, updated) + delivery latency`. Classifications add inference latency. Fills happen at
  the open of the first bar starting at or after the decision. Labels are only allowed to look ahead inside `research/`.
- Ruff bans `datetime.now()` everywhere except `core/clock.py`.

## Limitations and unfinished work

- **IEX-only data.** The free feed carries a small share of consolidated volume, so relative-volume features and fill
  prices are noisier than with SIP data. The cost model's spread and impact settings are estimates, not calibrated values.
- **No validated edge yet.** The pipeline and statistics exist, but no model has passed the promotion gate
  (OOS Rank IC > 0, HAC t > 2, DSR > 0.95). The rows in `legacy/trading.db` are contaminated (ticker missing from the
  prompt, bar lookahead) and must not be used for training. A new backfill of several months is needed.
- **Hosted and laya inference not exercised live.** Both adapters are tested against mocks or fakes only.
- **Equities only.** Options execution is out of scope for this rebuild.
- **Paper trading is plumbing-verified, not soak-tested.** The account check, a real order placement and cancellation,
  and a bounded runner session were run after the close. A full market-hours session, the two-week unattended run and
  the fill-vs-simulation deviation check still need to be done.
- **Ownership does not survive a restart.** On startup every existing position is treated as foreign, including ones a
  previous run of the runner opened, so they are never exited automatically afterwards. Stop with `--flatten-on-stop`,
  or close leftover positions manually before restarting.
- **`backtest --start/--end` is a single train/replay split**, not a rolling walk-forward of the replay. The model is fit
  once on the first `--train-fraction` of events (purged so no training label ends after the replay starts) and held fixed.
- **Promotion gate needs data.** It also requires at least 20 daily IC periods, because the HAC t-stat over a few days passed
  pure noise in 17.5% of 40 null simulations.
- **No monitoring or alerting yet.** Production-readiness items (Phase 8 of the plan) are not built.
