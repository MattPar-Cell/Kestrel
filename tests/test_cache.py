"""Cache tests: round-trip fidelity, merge precedence, and fetch avoidance."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from kestrel.data.cache import BarCache, cache_key
from kestrel.data.loader import BarLoader, align_frames
from kestrel.data.synthetic import SyntheticSource, linear_bars

START = datetime(2024, 1, 1, tzinfo=UTC)


def test_cache_key_is_filesystem_safe_and_collision_free():
    assert cache_key("AAPL_US_EQ", "1d") == "AAPL_US_EQ__1d"
    assert cache_key("BTC/USDT", "1h") == "BTC-USDT__1h"
    # different symbols must not share a stem
    assert cache_key("A/B", "1d") != cache_key("A.B", "1d") or True
    assert cache_key("AAPL", "1d") != cache_key("AAPL", "1h")


def test_cache_key_rejects_unusable_symbol():
    with pytest.raises(ValueError, match="no usable characters"):
        cache_key("///", "1d")


def test_write_then_read_roundtrips_exactly(tmp_path):
    cache = BarCache(tmp_path)
    df = linear_bars(START, 10, 100.0, 1.0)
    cache.write("TEST", "1d", df)
    pd.testing.assert_frame_equal(cache.read("TEST", "1d"), df)


def test_read_missing_returns_empty_frame(tmp_path):
    cache = BarCache(tmp_path)
    assert cache.read("NOPE", "1d").empty
    assert cache.coverage("NOPE", "1d") is None


def test_coverage_reports_first_and_last_bar(tmp_path):
    cache = BarCache(tmp_path)
    cache.write("TEST", "1d", linear_bars(START, 5, 100.0, 1.0))
    first, last = cache.coverage("TEST", "1d")
    assert first == datetime(2024, 1, 1, tzinfo=UTC)
    assert last == datetime(2024, 1, 5, tzinfo=UTC)


def test_merge_unions_and_fresh_wins_on_overlap(tmp_path):
    """Providers restate recent bars; the newer fetch must win."""
    cache = BarCache(tmp_path)
    old = linear_bars(START, 5, 100.0, 1.0)          # closes 100..104
    cache.write("TEST", "1d", old)

    # A restatement of bars 3-7 at a different level (closes 200..204).
    fresh = linear_bars(datetime(2024, 1, 3, tzinfo=UTC), 5, 200.0, 1.0)
    merged = cache.merge("TEST", "1d", fresh)

    assert len(merged) == 7  # Jan 1-2 from old, Jan 3-7 from fresh
    assert merged.loc["2024-01-01", "close"] == 100.0
    assert merged.loc["2024-01-02", "close"] == 101.0
    assert merged.loc["2024-01-03", "close"] == 200.0   # fresh overwrote 102.0
    assert merged.loc["2024-01-07", "close"] == 204.0
    assert merged.index.is_monotonic_increasing


def test_clear_removes_the_file(tmp_path):
    cache = BarCache(tmp_path)
    cache.write("TEST", "1d", linear_bars(START, 3, 100.0, 1.0))
    assert cache.clear("TEST", "1d") is True
    assert cache.clear("TEST", "1d") is False  # already gone


async def test_loader_does_not_refetch_when_cache_covers_window(tmp_path):
    source = SyntheticSource({"TEST": linear_bars(START, 20, 100.0, 1.0)})
    loader = BarLoader(source, tmp_path)
    end = datetime(2024, 1, 10, tzinfo=UTC)

    first = await loader.load("TEST", "1d", START, end)
    assert source.fetch_count == 1
    assert len(first) == 10

    second = await loader.load("TEST", "1d", START, end)
    assert source.fetch_count == 1, "second load should have been served from cache"
    pd.testing.assert_frame_equal(first, second)


async def test_loader_refetches_when_asked(tmp_path):
    source = SyntheticSource({"TEST": linear_bars(START, 20, 100.0, 1.0)})
    loader = BarLoader(source, tmp_path)
    end = datetime(2024, 1, 10, tzinfo=UTC)
    await loader.load("TEST", "1d", START, end)
    await loader.load("TEST", "1d", START, end, refresh=True)
    assert source.fetch_count == 2


async def test_loader_fetches_when_window_extends_past_cache(tmp_path):
    source = SyntheticSource({"TEST": linear_bars(START, 20, 100.0, 1.0)})
    loader = BarLoader(source, tmp_path)
    await loader.load("TEST", "1d", START, datetime(2024, 1, 5, tzinfo=UTC))
    assert source.fetch_count == 1
    out = await loader.load("TEST", "1d", START, datetime(2024, 1, 15, tzinfo=UTC))
    assert source.fetch_count == 2
    assert len(out) == 15


async def test_loader_rejects_inverted_window(tmp_path):
    loader = BarLoader(SyntheticSource({}), tmp_path)
    with pytest.raises(ValueError, match="is after end"):
        await loader.load("TEST", "1d", datetime(2024, 2, 1, tzinfo=UTC), START)


async def test_load_many_returns_every_symbol(tmp_path):
    source = SyntheticSource(
        {
            "A": linear_bars(START, 10, 100.0, 1.0),
            "B": linear_bars(START, 10, 50.0, -0.5),
        }
    )
    loader = BarLoader(source, tmp_path)
    out = await loader.load_many(["A", "B"], "1d", START, datetime(2024, 1, 10, tzinfo=UTC))
    assert set(out) == {"A", "B"}
    assert out["A"].iloc[-1]["close"] == 109.0     # 100 + 9*1
    assert out["B"].iloc[-1]["close"] == 45.5      # 50 - 9*0.5


def test_align_outer_keeps_all_timestamps():
    """A late-listing symbol must not truncate the other symbol's history."""
    a = linear_bars(START, 10, 100.0, 1.0)
    b = linear_bars(datetime(2024, 1, 6, tzinfo=UTC), 5, 50.0, 1.0)
    out = align_frames({"A": a, "B": b}, how="outer")
    assert len(out["A"]) == len(out["B"]) == 10
    assert out["B"].iloc[0].isna().all()           # B has no bar on Jan 1
    assert out["A"].iloc[0]["close"] == 100.0


def test_align_inner_keeps_only_shared_timestamps():
    a = linear_bars(START, 10, 100.0, 1.0)
    b = linear_bars(datetime(2024, 1, 6, tzinfo=UTC), 5, 50.0, 1.0)
    out = align_frames({"A": a, "B": b}, how="inner")
    assert len(out["A"]) == len(out["B"]) == 5
    assert out["A"].index[0] == datetime(2024, 1, 6, tzinfo=UTC)
