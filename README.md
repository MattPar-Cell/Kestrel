# Kestrel

A systematic trading bot, built in phases. Research and validation come first;
live execution is gated behind an explicit sign-off.

## Status

Phase 0 (scaffolding) complete. Nothing else is implemented.

| Phase | Scope | State |
|---|---|---|
| 0 | Project scaffolding, config, env template | done |
| 1 | Market data ingestion + event-driven backtest harness | done |
| 2 | First signal: EMA spread / ATR momentum | not started |
| 3 | Position sizing, portfolio constraints, metrics | not started |
| 4 | Telegram notifications (read-only) | not started |
| 5 | News/sentiment signal (optional) | not started |
| 6 | Paper trading | not started |

Live-money execution is not in this table. It happens only after Phase 6 results
are reviewed; `KESTREL_ENVIRONMENT=live` is rejected by config validation today.

## Layout

```
kestrel/
  data/       market + news ingestion (async feeds, parquet bar cache)
  features/   indicator/signal computation — pure functions, bars in / series out
  strategy/   signal generation logic
  risk/       position sizing, portfolio constraints, metrics
  execution/  broker order interface (paper/sandbox first)
  notify/     telegram
  backtest/   event-driven backtesting harness
  config/     settings + secrets loading (pydantic-settings)
tests/
```

The event-driven harness in `backtest/` shares code paths with `execution/`: a
strategy implements `Strategy.on_bars(ctx) -> list[OrderIntent]` and the same
method is called by the backtest engine and (in Phase 6) the live runner. The
vocabulary they share lives in `execution/types.py`. Keep anything that touches
a wall clock or a live socket out of `features/`, `strategy/`, and `risk/` —
those three stay pure and unit-testable.

### How the backtest avoids lookahead bias

The engine's loop, per bar `i`:

1. Fill orders decided at the close of bar `i-1`, at bar `i`'s **open**.
2. Mark the portfolio at bar `i`'s close, recording one equity point.
3. Hand the strategy history up to and including bar `i`'s close. Its orders
   queue for bar `i+1`'s open.

A strategy therefore cannot act on information it was not given — not because
strategies are written carefully, but because the only prices it ever sees are
closes at or before its decision point, and the only prices it ever trades at
are opens strictly after it. Two tests enforce this:

* `test_order_fills_at_the_next_bar_open_not_the_decision_close` — checks the
  exact equity curve against hand-computed values.
* `test_cheating_strategy_cannot_capture_the_jump_it_saw` — a strategy that buys
  the instant it sees a +50% jump fills at the post-jump open and earns exactly
  zero. If that test ever shows a profit, every backtest result in the repo is
  invalid.

Bars are timestamped by **close** time for the same reason: a bar stamped
`2024-01-02T00:00Z` contains only what was public by that instant.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then fill it in; .env is git-ignored
pytest
```

## Configuration

All settings come from environment variables prefixed `KESTREL_`, loaded and
validated by `kestrel/config/settings.py`. See `.env.example`.

Two things are deliberately *not* defaulted:

* **`KESTREL_SYMBOLS`** — the universe. You said you'd specify 3–5 liquid
  instruments.
* **`KESTREL_MAX_DAILY_DRAWDOWN` / `KESTREL_MAX_TOTAL_DRAWDOWN`** — the drawdown
  hard-stop. Left blank on purpose; `Settings.drawdown_stop_configured` reports
  whether a choice has been made, and Phase 3 should refuse to arm the stop
  until it has.

`KESTREL_RISK_PER_TRADE` defaults to 0.005 (0.5%) — the conservative end of the
0.5–1% range, not a silent pick in the middle.

## Broker: Trading 212

Three properties of Trading 212's public API shape this codebase. They are not
configuration details; each one changes what the strategy can be.

**1. There is no historical price endpoint.** The API covers account data,
instrument metadata, positions, orders, and history (filled orders, dividends,
cash transactions, CSV exports) — but no OHLCV bars or candles. Market data has
to come from a separate provider. That is why `data/` defines a `BarSource`
protocol that knows nothing about brokers, and `execution/` defines a `Broker`
protocol that knows nothing about bars. They are wired together only at the top
level. Trading 212's ticker format (`AAPL_US_EQ`) will not match the data
provider's (`AAPL`), so the Phase 6 adapter needs a mapping table.

**2. Invest and Stocks ISA accounts cannot short.** The public API is enabled
only for those account types, and neither supports short selling. A signed
momentum signal can therefore express long-or-flat, not long-or-short. This
roughly halves the opportunity set and makes results asymmetric in a downtrend
— a fact to build around in Phase 2, not discover in Phase 6. `allow_short`
defaults to `False` and the fill model enforces it: a sell is clamped to the
quantity actually held.

**3. Commission is zero, but FX is not.** There is no per-trade commission on
Invest/ISA. There is a 0.15% FX conversion fee whenever the instrument's
currency differs from the account's — which, for a GBP account trading US
equities, is every trade in both directions. About 30bps round-trip is the
dominant cost in the model, and it is what decides whether a daily-bar strategy
clears its own costs. List every non-base-currency instrument in
`KESTREL_FX_SYMBOLS`; config rejects a ticker that is not in the universe, so a
typo cannot quietly remove the fee.

Also worth knowing: the API is v0 beta, rate limits are strict and per-endpoint,
and **order endpoints are not idempotent** — a retried request can place a
second order. Phase 6's adapter needs its own de-duplication; retry-on-timeout
is not safe here.

Sources: [Trading 212 API docs](https://docs.trading212.com/api),
[rate limiting](https://docs.trading212.com/api/section/rate-limiting/how-it-works).

## What Phase 1 contains

```
kestrel/data/
  types.py       Bar, BarSource protocol, frame-contract validation
  synthetic.py   deterministic generators (constant/linear/step/random-walk)
  cache.py       parquet cache, atomic writes, fresh-wins merge
  loader.py      cache-aware concurrent loading, index alignment
kestrel/execution/
  types.py       Side, OrderIntent, Fill, Position, AccountState, Broker
kestrel/backtest/
  fills.py       slippage models, T212 cost structure, long-only enforcement
  portfolio.py   cash, FIFO lot matching, trade log
  engine.py      the event loop, Context, Strategy and RiskGate protocols
  walkforward.py rolling/anchored splits with purging
  runner.py      fold orchestration, stitched OOS curve, overfitting gap
kestrel/risk/
  metrics.py     Sharpe/Sortino/drawdown/VaR/correlation/trade stats
```

167 tests, all against synthetic data with hand-computed expected values. No
real market data has been fetched, and no strategy exists yet.

A few implementation choices worth knowing:

* **FIFO lot matching**, not average cost. Both give the same total P&L over a
  round trip, but different per-trade numbers — and per-trade numbers are what
  win rate and the Phase 3 Kelly estimate are built from.
* **Break-even counts as a loss.** A scratch trade paid costs for nothing.
* **Rolling walk-forward by default**, with a `purge` gap between train and test.
  Set `purge` to the longest indicator lookback, or the first test bars are
  predicted by features computed partly from training data.
* **`sharpe_se` is reported beside every Sharpe.** At one year of daily bars the
  standard error is about ±1.0, so a reported 1.5 is not distinguishable from
  0.5. `summary_text()` says so explicitly when the error exceeds the estimate.
* **Conservative fills throughout.** A limit order the bar merely touched fills
  at the limit, never better; a stop that gaps through fills at the gapped open,
  not at the stop price.

## Ground rules

* Risk/return trade-offs (sizing formulas, drawdown thresholds, signal
  parameters) get discussed before they get coded.
* Every strategy and risk function needs a unit test with a hand-checked
  expected value.
* The harness is tested against synthetic data with known outputs before any
  real data touches it.
* Backtest results that look too good get checked for lookahead bias and
  overfitting before they get believed.
