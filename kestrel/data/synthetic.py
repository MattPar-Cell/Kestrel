"""Deterministic synthetic bar sources.

These exist so the backtest harness can be verified against inputs whose
correct output can be worked out by hand. Every generator here is seeded and
reproducible — no wall clock, no RNG global state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from kestrel.data.types import BAR_COLUMNS, validate_bars

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def bar_index(start: datetime, periods: int, timeframe: str = "1d") -> pd.DatetimeIndex:
    """A regular UTC bar-close index. No weekend/holiday logic — synthetic only."""
    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    step = TIMEFRAME_DELTAS[timeframe]
    return pd.DatetimeIndex(
        [start + i * step for i in range(periods)], name="timestamp"
    ).tz_convert("UTC")


def _frame(
    index: pd.DatetimeIndex, close: np.ndarray, spread: float, volume: float
) -> pd.DataFrame:
    """Wrap a close path into a consistent OHLCV frame.

    The open of bar `i` is the close of bar `i-1` (continuous, no gaps), which
    makes next-bar-open fills exactly predictable in tests. High/low are the
    bar's own range widened by `spread`.
    """
    close = np.asarray(close, dtype="float64")
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    hi = np.maximum(open_, close) * (1.0 + spread)
    lo = np.minimum(open_, close) * (1.0 - spread)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": hi,
            "low": lo,
            "close": close,
            "volume": np.full_like(close, volume),
        },
        index=index,
    )[list(BAR_COLUMNS)]
    return validate_bars(df, "synthetic")


def constant_bars(
    start: datetime, periods: int, price: float = 100.0, timeframe: str = "1d"
) -> pd.DataFrame:
    """Flat price. A strategy run on this must produce zero P&L before costs."""
    idx = bar_index(start, periods, timeframe)
    return _frame(idx, np.full(periods, float(price)), spread=0.0, volume=1_000.0)


def linear_bars(
    start: datetime,
    periods: int,
    price: float = 100.0,
    step: float = 1.0,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Price rising (or falling) by a fixed amount per bar.

    Exact arithmetic, so expected equity curves are hand-computable: bar i has
    close = price + i*step and open = close of bar i-1.
    """
    idx = bar_index(start, periods, timeframe)
    close = price + step * np.arange(periods, dtype="float64")
    if (close <= 0).any():
        raise ValueError("linear_bars produced a non-positive price; reduce `step` or `periods`")
    return _frame(idx, close, spread=0.0, volume=1_000.0)


def step_bars(
    start: datetime,
    periods: int,
    price: float = 100.0,
    jump_at: int | None = None,
    jump_to: float = 150.0,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Flat, then a single jump. Useful for checking signal latency precisely."""
    idx = bar_index(start, periods, timeframe)
    close = np.full(periods, float(price))
    if jump_at is not None:
        if not 0 <= jump_at < periods:
            raise ValueError(f"jump_at {jump_at} out of range for {periods} periods")
        close[jump_at:] = float(jump_to)
    return _frame(idx, close, spread=0.0, volume=1_000.0)


def random_walk_bars(
    start: datetime,
    periods: int,
    price: float = 100.0,
    vol: float = 0.01,
    drift: float = 0.0,
    seed: int = 0,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Geometric random walk. Seeded, so runs are reproducible.

    `vol` and `drift` are per-bar log-return terms. This is for smoke-testing
    plumbing, never for judging a strategy — a walk has no edge to find, and any
    apparent edge on it is a bug or a fluke.
    """
    idx = bar_index(start, periods, timeframe)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=drift, scale=vol, size=periods)
    shocks[0] = 0.0
    close = float(price) * np.exp(np.cumsum(shocks))
    return _frame(idx, close, spread=vol / 2.0, volume=1_000.0)


class SyntheticSource:
    """A `BarSource` backed by a dict of pre-built frames."""

    name = "synthetic"

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = {s: validate_bars(df, s) for s, df in frames.items()}
        self.fetch_count = 0  # lets tests assert the cache prevented a refetch

    async def fetch(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        self.fetch_count += 1
        if symbol not in self._frames:
            raise KeyError(f"synthetic source has no symbol {symbol!r}")
        df = self._frames[symbol]
        return df.loc[(df.index >= start) & (df.index <= end)].copy()
