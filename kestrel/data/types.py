"""Core data types and the source protocol.

The `BarSource` protocol is deliberately separate from anything broker-shaped.
Trading 212's public API has no historical price endpoint, so bars come from a
different provider than orders do. Keeping them apart means the backtest never
depends on the broker being reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

#: Column schema every bar frame must satisfy, in order.
BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar.

    `timestamp` is the bar's **close** time, always tz-aware UTC. Using close
    time rather than open time is the convention that makes lookahead bias
    harder to write by accident: a bar stamped 2024-01-02T00:00Z contains only
    information that was public by that instant.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"{self.symbol}: bar timestamp must be tz-aware, got {self.timestamp!r}"
            )
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"{self.symbol} @ {self.timestamp}: inconsistent OHLC "
                f"(o={self.open} h={self.high} l={self.low} c={self.close})"
            )
        if self.volume < 0:
            raise ValueError(f"{self.symbol} @ {self.timestamp}: negative volume {self.volume}")


@runtime_checkable
class BarSource(Protocol):
    """Anything that can produce historical OHLCV bars.

    Implementations are async because real providers are network-bound; the
    synthetic and cached sources satisfy it trivially.
    """

    name: str

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return a validated bar frame for `symbol` over [start, end].

        Index: tz-aware UTC DatetimeIndex of bar close times, sorted, unique.
        Columns: exactly `BAR_COLUMNS`.
        """
        ...


def validate_bars(df: pd.DataFrame, symbol: str = "?") -> pd.DataFrame:
    """Assert the bar-frame contract and return the frame unchanged.

    Called on everything entering *and* leaving the cache. A malformed frame
    that reaches the backtest produces plausible-looking nonsense, which is far
    worse than an exception here.
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: bar frame missing columns {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{symbol}: bar frame index must be a DatetimeIndex, "
            f"got {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise ValueError(f"{symbol}: bar frame index must be tz-aware (UTC)")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: bar frame index must be sorted ascending")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"{symbol}: duplicate bar timestamps {dupes[:5]}")

    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        bad = df.index[ohlc.isna().any(axis=1)].tolist()
        raise ValueError(f"{symbol}: NaN in OHLC at {bad[:5]}")

    too_low = df[(df["open"] < df["low"]) | (df["close"] < df["low"])]
    too_high = df[(df["open"] > df["high"]) | (df["close"] > df["high"])]
    if len(too_low) or len(too_high):
        bad = sorted(set(too_low.index.tolist() + too_high.index.tolist()))
        raise ValueError(f"{symbol}: open/close outside high/low range at {bad[:5]}")
    if (df["high"] < df["low"]).any():
        bad = df.index[df["high"] < df["low"]].tolist()
        raise ValueError(f"{symbol}: high < low at {bad[:5]}")
    if (df["volume"] < 0).any():
        bad = df.index[df["volume"] < 0].tolist()
        raise ValueError(f"{symbol}: negative volume at {bad[:5]}")

    return df


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    """Build a validated bar frame from `Bar` objects."""
    if not bars:
        return empty_bar_frame()
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        },
        index=pd.DatetimeIndex([b.timestamp for b in bars], name="timestamp").tz_convert("UTC"),
    )
    return validate_bars(df.sort_index(), bars[0].symbol)


def frame_to_bars(df: pd.DataFrame, symbol: str) -> list[Bar]:
    """Inverse of `bars_to_frame`."""
    validate_bars(df, symbol)
    return [
        Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for ts, row in df.iterrows()
    ]


def empty_bar_frame() -> pd.DataFrame:
    """A correctly-typed empty bar frame."""
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in BAR_COLUMNS},
        index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
    )
