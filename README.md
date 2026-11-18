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
- **Paper trading is not yet soak-tested.** The two-week unattended paper run and the fill-deviation check from the
  plan still have to be done by running it.
- **CLI is fixture-only for now.** `optionsone backtest --fixture` runs the full pipeline (stub classifier ->
  point-in-time features -> purged walk-forward research on the first half -> out-of-sample replay on the second half)
  on a seeded synthetic market with a planted news drift. Backfill and paper-trading commands that wire `ingest` and
  `AlpacaPaperBroker` into the CLI are not built yet; the components are usable as a library.
- **Promotion gate needs data.** It also requires at least 20 daily IC periods, because the HAC t-stat over a few days passed
  pure noise in 17.5% of 40 null simulations.
- **No monitoring or alerting yet.** Production-readiness items (Phase 8 of the plan) are not built.
