"""Local parquet cache for OHLCV bars.

One file per (symbol, timeframe). Repeated backtests read from disk instead of
re-hitting a provider, which matters both for speed and for not burning a rate
limit during research.

The cache is deliberately dumb about *what* a bar means — it validates the
frame contract and nothing else. Adjustment policy (splits, dividends) belongs
to the source, because mixing adjusted and unadjusted bars in one cache file
would silently corrupt backtests.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from kestrel.data.types import empty_bar_frame, validate_bars

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def cache_key(symbol: str, timeframe: str) -> str:
    """Filesystem-safe stem for a symbol/timeframe pair.

    `BTC/USDT` and `AAPL_US_EQ` both have to become one sane filename, and two
    different symbols must never collide onto the same stem.
    """
    safe = _UNSAFE.sub("-", symbol.strip()).strip("-")
    if not safe:
        raise ValueError(f"symbol {symbol!r} has no usable characters for a cache key")
    return f"{safe}__{timeframe}"


class BarCache:
    """Read/write/merge bar frames on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{cache_key(symbol, timeframe)}.parquet"

    def has(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()

    def read(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Read the whole cached history, or an empty frame if absent."""
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return empty_bar_frame()
        df = pd.read_parquet(path)
        if df.index.tz is None:  # parquet round-trips tz, but be defensive
            df.index = df.index.tz_localize("UTC")
        return validate_bars(df.sort_index(), symbol)

    def read_range(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        df = self.read(symbol, timeframe)
        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        """Replace the cached history for this symbol/timeframe."""
        validate_bars(df, symbol)
        path = self.path_for(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", index=True)
        tmp.replace(path)  # atomic, so an interrupted write can't leave a half file
        return path

    def merge(self, symbol: str, timeframe: str, fresh: pd.DataFrame) -> pd.DataFrame:
        """Union cached and fresh bars, with `fresh` winning on overlap.

        Fresh-wins is the right precedence: providers restate recent bars (late
        prints, adjustments), and the newer fetch is the more accurate one.
        """
        validate_bars(fresh, symbol)
        existing = self.read(symbol, timeframe)
        if existing.empty:
            merged = fresh.copy()
        elif fresh.empty:
            merged = existing
        else:
            merged = pd.concat([existing, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.write(symbol, timeframe, merged)
        return merged

    def coverage(self, symbol: str, timeframe: str) -> tuple[datetime, datetime] | None:
        """(first, last) cached bar timestamp, or None if nothing is cached."""
        df = self.read(symbol, timeframe)
        if df.empty:
            return None
        return df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()

    def clear(self, symbol: str, timeframe: str) -> bool:
        path = self.path_for(symbol, timeframe)
        if path.exists():
            path.unlink()
            return True
        return False
