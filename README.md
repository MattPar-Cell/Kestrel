# Kestrel

A systematic trading bot, built in phases. Research and validation come first;
live execution is gated behind an explicit sign-off.

## Status

Phase 0 (scaffolding) complete. Nothing else is implemented.

| Phase | Scope | State |
|---|---|---|
| 0 | Project scaffolding, config, env template | done |
| 1 | Market data ingestion + event-driven backtest harness | not started |
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

The event-driven harness in `backtest/` is meant to share code paths with
`execution/`, so the same signal and risk calls run in both. Keep anything that
touches a wall clock or a live socket out of `features/`, `strategy/`, and
`risk/` — those three stay pure and unit-testable.

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

## Broker SDK

`ccxt` is listed as a placeholder so the dependency set is complete, and
`KESTREL_EXCHANGE_ID` wires to it. It is the right choice only if the target is
crypto exchanges. For US equities/futures it is the wrong tool — that would be
an Alpaca / IBKR / Tradier SDK instead, and the asset class also changes the bar
calendar, the ATR units, and whether fractional position sizes are even legal.
No code depends on `ccxt` yet, so swapping it costs nothing right now.

## Ground rules

* Risk/return trade-offs (sizing formulas, drawdown thresholds, signal
  parameters) get discussed before they get coded.
* Every strategy and risk function needs a unit test with a hand-checked
  expected value.
* The harness is tested against synthetic data with known outputs before any
  real data touches it.
* Backtest results that look too good get checked for lookahead bias and
  overfitting before they get believed.
