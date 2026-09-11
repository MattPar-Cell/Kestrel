"""Cache-aware bar loading across a universe of symbols.

`BarLoader` is the only thing the backtest talks to. It decides whether a fetch
is needed, fans out concurrently across symbols, merges into the cache, and
hands back aligned frames.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from kestrel.data.cache import BarCache
from kestrel.data.types import BarSource, validate_bars

log = logging.getLogger(__name__)


class BarLoader:
    def __init__(
        self,
        source: BarSource,
        cache_dir: Path,
        max_concurrency: int = 4,
    ) -> None:
        """`max_concurrency` caps parallel provider calls.

        Kept low by default: every free market-data provider rate-limits, and a
        burst of 20 concurrent requests is the fastest way to get throttled
        mid-backtest.
        """
        self.source = source
        self.cache = BarCache(cache_dir)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return bars for [start, end], fetching only what the cache lacks.

        With `refresh=False` (the default) a cache that already spans the
        requested window is used as-is and the provider is never called — that's
        what makes repeated backtests cheap.
        """
        if start > end:
            raise ValueError(f"{symbol}: start {start} is after end {end}")

        if not refresh and self._cache_covers(symbol, timeframe, start, end):
            log.debug("%s %s: cache hit", symbol, timeframe)
            return self.cache.read_range(symbol, timeframe, start, end)

        async with self._sem:
            log.info(
                "%s %s: fetching %s → %s from %s",
                symbol, timeframe, start, end, self.source.name,
            )
            fresh = await self.source.fetch(symbol, timeframe, start, end)
        validate_bars(fresh, symbol)
        self.cache.merge(symbol, timeframe, fresh)
        return self.cache.read_range(symbol, timeframe, start, end)

    async def load_many(
        self,
        symbols: list[str] | tuple[str, ...],
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load several symbols concurrently. One failure fails the whole load.

        Silently dropping a symbol would change the universe a backtest ran on
        without saying so, which is the kind of quiet difference that makes two
        runs incomparable.
        """
        results = await asyncio.gather(
            *(self.load(s, timeframe, start, end, refresh=refresh) for s in symbols)
        )
        return dict(zip(symbols, results, strict=True))

    def _cache_covers(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> bool:
        cov = self.cache.coverage(symbol, timeframe)
        if cov is None:
            return False
        first, last = cov
        return first <= start and last >= end


def align_frames(
    frames: dict[str, pd.DataFrame], *, how: str = "outer"
) -> dict[str, pd.DataFrame]:
    """Reindex every frame onto a shared timestamp index.

    `how="outer"` keeps every timestamp any symbol traded on and leaves NaN
    where a symbol has no bar — the engine then simply has no bar event for that
    symbol at that time. `how="inner"` keeps only fully-populated timestamps.

    Outer is the default because inner silently discards history: one instrument
    with a late listing date would truncate the entire backtest to its lifetime.
    """
    if not frames:
        return {}
    if how not in {"inner", "outer"}:
        raise ValueError(f"how must be 'inner' or 'outer', got {how!r}")

    indexes = [df.index for df in frames.values()]
    shared = indexes[0]
    for idx in indexes[1:]:
        shared = shared.intersection(idx) if how == "inner" else shared.union(idx)
    shared = shared.sort_values()
    return {s: df.reindex(shared) for s, df in frames.items()}
