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

1. **A market-data adapter.** Only synthetic bars exist. Webull's Market Data
   API can supply both stocks and crypto.
2. **A strategy (Phase 2)** and a **sizer + risk gate (Phase 3)**. "Buy $X of Y"
   is exactly what these two produce. Until they exist and have been
   walk-forward tested, a recommendation is a number with nothing behind it.
3. **A read-only Webull adapter** — account → `Snapshot`. The tracker is built
   from it.
4. **A scheduler**: a daily job (decide after close → Signal message; append a
   snapshot) and a weekly job (summary → Signal). Cron or a systemd timer is
   enough; the job must run somewhere that is always on.
5. **The drawdown hard-stop choice** (still deliberately blank).
6. **Crypto's 24/7 calendar.** The fee model knows Webull's 1% crypto spread, but
   annualisation and the "next open" timing in messages assume a market that
   closes. Crypto does not.

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

## Broker: Webull

One Webull account holds stocks, ETFs, and crypto, behind one App Key / App
Secret pair. Four of its properties shape this codebase.

**1. What the API reaches depends on your Webull region.** Each region (US, HK,
SG, AU, JP, MY, BR) is a separate company with its own API host and product
list. A **US** account reaches stocks, ETFs, options, and crypto through the
API. Malaysia's API (launched July 2026) covers US stocks and ETFs only. Check
your region's developer portal before counting on crypto. Set
`KESTREL_BROKER_REGION`.

**2. Crypto costs 1% each side.** Webull charges no crypto commission but quotes
1% away from the mid price on every buy and sell, so it costs the same as a
fee. About 2% round trip, against zero commission on US stocks. That gap
decides what the crypto half of a strategy can be: anything that turns crypto
over weekly will not clear it. List crypto instruments in
`KESTREL_CRYPTO_SYMBOLS` so the backtest charges it.

**3. The defaults are the US fee schedule.** Zero stock commission, no
per-trade currency conversion (the account is in USD; your local currency is
converted once, on deposit), and a USD 5 fractional-order minimum. Other
regions charge commission and platform fees on stocks: put the all-in rate in
`KESTREL_COMMISSION_BPS`, or every backtest will look better than it is.

**4. Long-or-flat.** Shorting needs a margin account and is impossible for
crypto, so Kestrel assumes a cash account. `allow_short` defaults to `False`
and the fill model clamps a sell to the quantity held. That halves the
opportunity set and makes results lopsided in a downtrend; build around it in
Phase 2.

Webull's Market Data API serves price bars too, so one key pair can cover
prices as well as the account. `BarSource` and `Broker` stay separate protocols
anyway, so a backtest never needs the broker to be reachable.

## Setup: keys and Signal

Kestrel's default mode is **advisory**: it messages you and *you* place the
trade in the Webull app. The bot only ever reads your account.

**Webull API key.** The API is not on by default; you apply for it.

1. Open a Webull account and finish identity verification. The API
   application is only possible once the account is open.
2. On the Webull **website** (not the app), go to **OpenAPI Management → My
   Application** and apply, saying it is for personal portfolio tracking and
   trade alerts. Webull reviews it, typically in 1–2 business days.
3. Once approved: **OpenAPI Management → App Management**, register an app,
   then **Generate Key**. It asks for an SMS code and your trading password.
4. Put the **App Key** in `KESTREL_BROKER_API_KEY` and the **App Secret** in
   `KESTREL_BROKER_API_SECRET`.

The App Secret signs requests and can place orders, so treat it like your
trading password: never commit it, never paste it into chats or issues, and
regenerate it if it leaks. If your region's portal offers IP whitelisting, use
it.

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
Portfolio $10,000.00 · cash $2,500.00 (25.0%)

1. BUY AAPL
   $1,000.00 (10.0% of portfolio) ≈ 5 @ $200.00
   When: Mon 05 Oct 09:30–10:30 (America/New_York)
   Stop: $185.00
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
  fills.py       slippage models, Webull cost structure, long-only enforcement
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
