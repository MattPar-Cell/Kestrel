# Kestrel

A systematic trading bot, built in phases. Research and validation come first;
live execution is gated behind an explicit sign-off.

## Status

Phases 0–1 are done. Phase 4's building blocks (Signal transport, message text,
portfolio tracker, weekly summary) are in, but nothing generates trades yet.

| Phase | Scope | State |
|---|---|---|
| 0 | Project scaffolding, config, env template | done |
| 1 | Market data ingestion + event-driven backtest harness | done |
| 2 | First signal: EMA spread / ATR momentum | not started |
| 3 | Position sizing, portfolio constraints, metrics | not started |
| 4 | Signal notifications + portfolio tracker | building blocks done |
| 5 | News/sentiment signal (optional) | not started |
| 6 | Paper trading | not started |

### What is still missing before a Signal message means anything

1. **A market-data adapter.** Only synthetic bars exist. Stocks need a provider
   (Trading 212 has no price history); crypto can use Kraken's public OHLC.
2. **A strategy (Phase 2)** and a **sizer + risk gate (Phase 3)**. "Buy £X of Y"
   is exactly what these two produce. Until they exist and have been
   walk-forward tested, a recommendation is a number with nothing behind it.
3. **Read-only venue adapters** — Trading 212 and Kraken → `Snapshot`, plus a
   GBP conversion for crypto balances. The tracker is built from these.
4. **A scheduler**: a daily job (decide after close → Signal message; append a
   snapshot) and a weekly job (summary → Signal). Cron or a systemd timer is
   enough; the job must run somewhere that is always on.
5. **The drawdown hard-stop choice** (still deliberately blank).
6. **A crypto cost model.** The fill model knows Trading 212's FX fee, not
   Kraken's maker/taker fees or crypto's 24/7 sessions.

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
  notify/     Signal messages: recommendation + weekly summary text, transport
  tracker/    portfolio snapshots, snapshot log, weekly summary
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

## Crypto: Kraken, alongside Trading 212

Trading 212's public API reaches Invest and Stocks ISA only; its crypto product
is not exposed through it. Crypto therefore lives at a second venue, and Kestrel
treats the two as separate: separate key pairs (`BROKER_*` and `CRYPTO_*`),
separate adapters, one combined `Snapshot` in GBP. Kraken is the choice because
it is FCA-registered, has a long-standing REST API with per-key permission
scopes, and serves free public OHLC history, so it covers crypto data too.

## Setup: keys and Signal

Kestrel's default mode is **advisory**: it messages you and *you* place the
trade. In that mode no key needs permission to trade — read scopes are enough,
and a leaked key cannot move money. Only Phase 6 automated execution would need
order scopes, and then only on the demo key first.

**Trading 212** (app → Settings → API (Beta) → Generate API key). Create it on
the *practice* account first; demo and live keys are separate. Enable
account data, portfolio, history, and metadata scopes; leave order execution
and pies-write off. Restrict it to your server's IP if you can. The secret is
shown once — put both values in `.env` as `KESTREL_BROKER_API_KEY` /
`KESTREL_BROKER_API_SECRET`.

**Kraken** (Settings → API → Create API key). Tick *Query Funds*, *Query Open
Orders & Trades*, *Query Closed Orders & Trades*. Leave *Create & Modify
Orders*, *Cancel/Close Orders*, and above all **Withdraw Funds** unticked. Store
as `KESTREL_CRYPTO_API_KEY` / `KESTREL_CRYPTO_API_SECRET`.

**Signal.** Signal has no bot API; run the `bbernhard/signal-cli-rest-api`
container and link it to a Signal account as a secondary device. A spare number
is cleanest — then the messages arrive as from someone else rather than in your
"Note to self".

```bash
docker run -d --name signal-api --restart=always -p 127.0.0.1:8080:8080 \
  -v $HOME/.local/share/signal-api:/home/.local/share/signal-cli \
  -e MODE=native bbernhard/signal-cli-rest-api
# open http://localhost:8080/v1/qrcodelink?device_name=kestrel and scan it from
# Signal → Settings → Linked devices, then test:
curl -X POST localhost:8080/v2/send -H 'Content-Type: application/json' \
  -d '{"message":"kestrel test","number":"+44...","recipients":["+44..."]}'
```

Bind it to `127.0.0.1` as above: anyone who can reach that port can send
messages as you. Then set `KESTREL_SIGNAL_SENDER` and
`KESTREL_SIGNAL_RECIPIENTS`.

### What the messages look like

A recommendation says what, how much (as money *and* as a share of the
portfolio now), and a window after which to ignore it:

```
Kestrel: 1 trade
Portfolio £10,000.00 · cash £2,500.00 (25.0%)

1. BUY AAPL_US_EQ
   £1,000.00 (10.0% of portfolio) ≈ 5 @ £200.00
   When: Mon 05 Oct 14:30–15:30 (Europe/London)
   Stop: £185.00
   Why: ema_cross_up

Skip any trade you see after its window closes.
```

The weekly summary reports P&L with deposits excluded, the worst dip of the
week, trades and fees, cash, holdings by weight, and the biggest movers.

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
