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
* `allow_short` defaults to False. Trading 212 Invest and Stocks ISA accounts
  cannot sell short, so a signed signal can only ever express long-or-flat.
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
    #: Trading 212 instrument tickers, e.g. "AAPL_US_EQ". Its API uses its own
    #: ticker format, which does NOT match the market-data provider's — the
    #: Phase 6 adapter needs a mapping table, not string equality.
    symbols: tuple[str, ...] = ()
    timeframe: str = "1d"
    data_cache_dir: Path = REPO_ROOT / "data_cache"

    #: Trading 212 has no historical price endpoint, so bars come from elsewhere.
    market_data_provider: str = "csv"

    # ---- broker: Trading 212 --------------------------------------------
    #: Account base currency. Every instrument priced in anything else incurs
    #: the FX fee on both legs of every trade.
    base_currency: str = "GBP"
    #: Invest and Stocks ISA cannot short. Leave False unless the broker changes.
    allow_short: bool = False
    #: Instruments not denominated in `base_currency`, which therefore pay the
    #: FX fee. For a GBP account trading US stocks, this is all of them.
    fx_symbols: tuple[str, ...] = ()

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
    #: Zero on Trading 212 Invest/ISA — there is no per-trade commission.
    commission_bps: float = Field(default=0.0, ge=0)
    #: Trading 212's 0.15% FX conversion fee.
    fx_fee_bps: float = Field(default=15.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    #: Orders below this notional are rejected by the broker.
    min_order_value: float = Field(default=1.0, ge=0)

    # ---- secrets ---------------------------------------------------------
    broker_api_key: SecretStr | None = None
    broker_api_secret: SecretStr | None = None
    news_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: int | None = None

    # ---- validation ------------------------------------------------------
    @field_validator("symbols", "fx_symbols", mode="before")
    @classmethod
    def _split_symbols(cls, v: object) -> object:
        """Accept `KESTREL_SYMBOLS=AAPL_US_EQ,MSFT_US_EQ` as well as a real list."""
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
    def _fx_symbols_are_in_the_universe(self) -> Settings:
        """A typo in `fx_symbols` would silently understate costs by 15bps a side."""
        if self.symbols:
            unknown = set(self.fx_symbols) - set(self.symbols)
            if unknown:
                raise ValueError(
                    f"fx_symbols contains symbols not in the universe: {sorted(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def _paper_needs_broker_creds(self) -> Settings:
        if self.environment is Environment.PAPER and not self.broker_api_key:
            raise ValueError("environment='paper' requires KESTREL_BROKER_API_KEY")
        return self

    # ---- derived ---------------------------------------------------------
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
