"""Bar-frame contract tests. A malformed frame must raise, never pass quietly."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from kestrel.data.synthetic import linear_bars
from kestrel.data.types import Bar, bars_to_frame, frame_to_bars, validate_bars

START = datetime(2024, 1, 1, tzinfo=UTC)


def test_valid_bar_roundtrips_through_frame():
    df = linear_bars(START, 5, 100.0, 1.0)
    bars = frame_to_bars(df, "TEST")
    assert len(bars) == 5
    # close of bar i is 100 + i; open of bar i is close of bar i-1
    assert [b.close for b in bars] == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert [b.open for b in bars] == [100.0, 100.0, 101.0, 102.0, 103.0]
    rebuilt = bars_to_frame(bars)
    pd.testing.assert_frame_equal(df, rebuilt)


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        Bar("X", datetime(2024, 1, 1), 100, 101, 99, 100, 1000)


def test_close_outside_high_low_is_rejected():
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        Bar("X", START, open=100, high=101, low=99, close=105, volume=1000)


def test_negative_volume_is_rejected():
    with pytest.raises(ValueError, match="negative volume"):
        Bar("X", START, 100, 101, 99, 100, -1)


def test_unsorted_index_is_rejected():
    df = linear_bars(START, 3, 100.0, 1.0).iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_bars(df, "X")


def test_duplicate_timestamps_are_rejected():
    df = linear_bars(START, 3, 100.0, 1.0)
    dupe = pd.concat([df, df.iloc[[1]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(dupe, "X")


def test_naive_frame_index_is_rejected():
    df = linear_bars(START, 3, 100.0, 1.0)
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        validate_bars(df, "X")


def test_nan_in_ohlc_is_rejected():
    df = linear_bars(START, 3, 100.0, 1.0)
    df.iloc[1, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(ValueError, match="NaN in OHLC"):
        validate_bars(df, "X")


def test_high_below_low_is_rejected():
    df = linear_bars(START, 3, 100.0, 1.0)
    df.iloc[1, df.columns.get_loc("high")] = 50.0
    with pytest.raises(ValueError, match="outside high/low|high < low"):
        validate_bars(df, "X")


def test_missing_column_is_rejected():
    df = linear_bars(START, 3, 100.0, 1.0).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_bars(df, "X")
