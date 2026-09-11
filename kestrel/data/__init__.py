"""Market and news data ingestion (async feeds + local parquet cache)."""

from kestrel.data.cache import BarCache, cache_key
from kestrel.data.loader import BarLoader, align_frames
from kestrel.data.types import (
    BAR_COLUMNS,
    Bar,
    BarSource,
    bars_to_frame,
    empty_bar_frame,
    frame_to_bars,
    validate_bars,
)

__all__ = [
    "BAR_COLUMNS",
    "Bar",
    "BarCache",
    "BarLoader",
    "BarSource",
    "align_frames",
    "bars_to_frame",
    "cache_key",
    "empty_bar_frame",
    "frame_to_bars",
    "validate_bars",
]
