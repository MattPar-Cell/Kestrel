"""Typed, validated configuration for Kestrel.

Everything the bot needs to run comes from environment variables (or a local
`.env` file, which is git-ignored). Nothing here reads from the network, and
nothing here has a side effect at import time other than building the object.

Design notes worth knowing before Phase 3:

* `risk_per_trade` defaults to the *conservative* end of the range we discussed
  (0.5%). It is a real risk/return knob, so it is surfaced here rather than
  buried in the sizer.
* `max_daily_drawdown` / `max_total_drawdown` deliberately default to `None`.
  A drawdown hard-stop is the single most consequential switch in the system —
  it can take the strategy flat at the worst possible moment — so it must be
  set explicitly rather than inherited from a default. Phase 3 code should
  refuse to arm the hard-stop while these are `None`.
* Kelly sizing is off by default (`sizing_mode="vol_target"`).
* `allow_short` defaults to False. Shorting needs a Webull margin account and
  is impossible for crypto, so Kestrel assumes a cash account: long-or-flat.
* Stocks, ETFs, and crypto all sit in one Webull account behind one App Key /
  App Secret pair. What the API can reach depends on `broker_region`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    """Which execution surface we are pointed at.

    `LIVE` exists as a value so config can *reject* it until the user explicitly
    signs off on Phase 6; no live-execution code path consumes it yet.
    """

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class SizingMode(StrEnum):
    VOL_TARGET = "vol_target"      # account_risk_per_trade / (ATR * multiplier)
    FRACTIONAL_KELLY = "fractional_kelly"  # Phase 3, opt-in only


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KESTREL_",
        extra="ignore",
        frozen=True,
    )

    # ---- environment -----------------------------------------------------
    environment: Environment = Environment.BACKTEST

    # ---- universe & data -------------------------------------------------
    #: Webull symbols, e.g. "AAPL", "BTCUSD". The API trades by instrument id,
    #: so the adapter resolves each symbol once and caches it.
    symbols: tuple[str, ...] = ()
    timeframe: str = "1d"
    data_cache_dir: Path = REPO_ROOT / "data_cache"

    #: Where bars come from. Webull's Market Data API serves bars; "csv" until
    #: that adapter exists.
    market_data_provider: str = "csv"

    # ---- broker: Webull --------------------------------------------------
    #: Which Webull entity holds the account. Each region has its own API host
    #: and its own product list — US is the one with crypto via the API.
    broker_region: str = "us"
    #: Account base currency. Every instrument priced in anything else incurs
    #: the FX fee on both legs of every trade.
    base_currency: str = "USD"
    #: Requires a margin account, and never applies to crypto. Leave False.
    allow_short: bool = False
    #: Instruments that convert currency on every trade, so pay the FX fee both
    #: ways. Usually empty: a USD account trading US stocks converts nothing.
    fx_symbols: tuple[str, ...] = ()
    #: Instruments that pay Webull's crypto spread, e.g. "BTCUSD,ETHUSD".
    crypto_symbols: tuple[str, ...] = ()

    # ---- risk (see module docstring) -------------------------------------
    account_equity: float = Field(default=100_000.0, gt=0)
    risk_per_trade: float = Field(default=0.005, gt=0, le=0.05)
    atr_multiplier: float = Field(default=2.0, gt=0)
    max_concurrent_positions: int = Field(default=3, ge=1)
    max_pairwise_correlation: float = Field(default=0.7, ge=-1.0, le=1.0)
    max_daily_drawdown: float | None = Field(default=None, gt=0, lt=1)
    max_total_drawdown: float | None = Field(default=None, gt=0, lt=1)
    sizing_mode: SizingMode = SizingMode.VOL_TARGET
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)

    # ---- backtest fill model --------------------------------------------
    #: Zero on Webull US for stocks and ETFs. Other regions charge — set it.
    commission_bps: float = Field(default=0.0, ge=0)
    #: Webull's 1% crypto spread, each side.
    crypto_fee_bps: float = Field(default=100.0, ge=0)
    #: Per-trade conversion fee for `fx_symbols`. Zero for a USD account.
    fx_fee_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    #: Orders below this notional are rejected by the broker (Webull: USD 5).
    min_order_value: float = Field(default=5.0, ge=0)

    # ---- notifications: Signal ------------------------------------------
    #: Base URL of a signal-cli-rest-api instance you run yourself. Signal has no
    #: official bot API; this container holds a linked Signal device.
    signal_api_url: str = "http://localhost:8080"
    #: The Signal number the bot sends *from* (E.164, e.g. "+447700900123").
    signal_sender: str | None = None
    #: Who receives messages: E.164 numbers or Signal group ids.
    signal_recipients: tuple[str, ...] = ()
    #: Times in messages are shown in this zone.
    #: Set it to where you live, e.g. "America/New_York", "Asia/Kuala_Lumpur".
    display_timezone: str = "UTC"
    #: Where the portfolio tracker appends its daily snapshots.
    snapshot_log: Path = REPO_ROOT / "data_cache" / "snapshots.jsonl"

    # ---- secrets ---------------------------------------------------------
    #: Webull App Key and App Secret, from OpenAPI Management on the Webull
    #: website once the API application is approved. Requests are signed with
    #: the secret; it never leaves this process.
    broker_api_key: SecretStr | None = None
    broker_api_secret: SecretStr | None = None
    news_api_key: SecretStr | None = None

    # ---- validation ------------------------------------------------------
    @field_validator(
        "symbols", "fx_symbols", "crypto_symbols", "signal_recipients", mode="before"
    )
    @classmethod
    def _split_symbols(cls, v: object) -> object:
        """Accept `KESTREL_SYMBOLS=AAPL,BTC` as well as a real list."""
        if isinstance(v, str):
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return v

    @field_validator("timeframe")
    @classmethod
    def _known_timeframe(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"timeframe must be one of {sorted(allowed)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _live_requires_explicit_signoff(self) -> Settings:
        if self.environment is Environment.LIVE:
            raise ValueError(
                "environment='live' is not supported yet. Live execution is gated "
                "behind Phase 6 sign-off after reviewing paper-trading results."
            )
        return self

    @field_validator("base_currency")
    @classmethod
    def _currency_code(cls, v: str) -> str:
        code = v.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"base_currency must be a 3-letter ISO code, got {v!r}")
        return code

    @model_validator(mode="after")
    def _fee_lists_are_in_the_universe(self) -> Settings:
        """A typo in a fee list would silently drop that fee from the backtest."""
        if self.symbols:
            for name in ("fx_symbols", "crypto_symbols"):
                unknown = set(getattr(self, name)) - set(self.symbols)
                if unknown:
                    raise ValueError(
                        f"{name} contains symbols not in the universe: {sorted(unknown)}"
                    )
        return self

    @field_validator("broker_region")
    @classmethod
    def _known_region(cls, v: str) -> str:
        """Each Webull region is a separate entity with its own API host."""
        allowed = {"us", "hk", "sg", "au", "jp", "my", "br"}
        region = v.strip().lower()
        if region not in allowed:
            raise ValueError(f"broker_region must be one of {sorted(allowed)}, got {v!r}")
        return region

    @field_validator("display_timezone")
    @classmethod
    def _known_timezone(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"display_timezone must be an IANA zone name, got {v!r}") from e
        return v

    @field_validator("signal_sender")
    @classmethod
    def _e164(cls, v: str | None) -> str | None:
        """signal-cli addresses accounts by international number; a local one fails late."""
        if v is not None and not (v.startswith("+") and v[1:].isdigit()):
            raise ValueError(f"signal_sender must be an E.164 number like +447700900123, got {v!r}")
        return v

    @model_validator(mode="after")
    def _paper_needs_broker_creds(self) -> Settings:
        if self.environment is Environment.PAPER and not (
            self.broker_api_key and self.broker_api_secret
        ):
            raise ValueError(
                "environment='paper' requires KESTREL_BROKER_API_KEY and "
                "KESTREL_BROKER_API_SECRET (a Webull App Key / App Secret pair)"
            )
        return self

    # ---- derived ---------------------------------------------------------
    @property
    def signal_configured(self) -> bool:
        return self.signal_sender is not None and bool(self.signal_recipients)

    @property
    def drawdown_stop_configured(self) -> bool:
        """True once the operator has made a deliberate drawdown-stop choice."""
        return self.max_daily_drawdown is not None or self.max_total_drawdown is not None

    @property
    def risk_budget_per_trade(self) -> float:
        """Currency amount at risk on a single trade."""
        return self.account_equity * self.risk_per_trade


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
