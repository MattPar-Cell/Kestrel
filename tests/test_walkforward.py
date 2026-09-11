"""Walk-forward window arithmetic, checked by hand."""

import pytest

from kestrel.backtest.walkforward import (
    SplitScheme,
    Window,
    slice_frames,
    walk_forward_splits,
)
from kestrel.data.synthetic import linear_bars
from tests.conftest import START


def test_window_basics():
    w = Window(10, 20)
    assert len(w) == 10
    df = linear_bars(START, 30, 100.0, 1.0)
    assert len(w.slice(df)) == 10
    assert w.slice(df).iloc[0]["close"] == 110.0  # bar 10 close = 100 + 10


def test_empty_window_is_rejected():
    with pytest.raises(ValueError, match="empty window"):
        Window(10, 10)


def test_rolling_splits_have_contiguous_non_overlapping_test_windows():
    """100 bars, train 50, test 10, step defaults to test_size.

    fold 0: train [0,50)  test [50,60)
    fold 1: train [10,60) test [60,70)
    ...
    fold 4: train [40,90) test [90,100)
    """
    folds = walk_forward_splits(100, train_size=50, test_size=10)
    assert len(folds) == 5
    assert [(f.train.start, f.train.stop) for f in folds] == [
        (0, 50), (10, 60), (20, 70), (30, 80), (40, 90)
    ]
    assert [(f.test.start, f.test.stop) for f in folds] == [
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)
    ]
    # every train window is the same length, and test windows tile exactly
    assert {len(f.train) for f in folds} == {50}
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test.stop == b.test.start


def test_anchored_splits_keep_the_start_and_grow_the_train_window():
    folds = walk_forward_splits(100, train_size=50, test_size=10, scheme=SplitScheme.ANCHORED)
    assert all(f.train.start == 0 for f in folds)
    assert [len(f.train) for f in folds] == [50, 60, 70, 80, 90]
    assert [(f.test.start, f.test.stop) for f in folds] == [
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)
    ]


def test_validate_window_sits_between_train_and_test():
    folds = walk_forward_splits(100, train_size=50, test_size=10, validate_size=10)
    f = folds[0]
    assert (f.train.start, f.train.stop) == (0, 50)
    assert (f.validate.start, f.validate.stop) == (50, 60)
    assert (f.test.start, f.test.stop) == (60, 70)
    # validation never overlaps the test window it selects for
    assert f.validate.stop <= f.test.start


def test_purge_gaps_every_boundary():
    """purge=5 drops 5 bars after train and 5 more after validate."""
    folds = walk_forward_splits(100, train_size=50, test_size=10, validate_size=10, purge=5)
    f = folds[0]
    assert (f.train.start, f.train.stop) == (0, 50)
    assert (f.validate.start, f.validate.stop) == (55, 65)   # 50 + 5
    assert (f.test.start, f.test.stop) == (70, 80)           # 65 + 5
    assert f.test.start - f.validate.stop == 5


def test_no_validation_window_when_size_is_zero():
    assert walk_forward_splits(100, train_size=50, test_size=10)[0].validate is None


def test_custom_step_can_overlap_test_windows():
    folds = walk_forward_splits(100, train_size=50, test_size=10, step=5)
    assert [(f.test.start, f.test.stop) for f in folds][:3] == [(50, 60), (55, 65), (60, 70)]


def test_insufficient_history_yields_no_folds():
    assert walk_forward_splits(40, train_size=50, test_size=10) == []
    assert walk_forward_splits(59, train_size=50, test_size=10) == []
    assert len(walk_forward_splits(60, train_size=50, test_size=10)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_size": 0, "test_size": 10},
        {"train_size": 50, "test_size": 0},
        {"train_size": 50, "test_size": 10, "step": 0},
        {"train_size": 50, "test_size": 10, "purge": -1},
        {"train_size": 50, "test_size": 10, "validate_size": -1},
    ],
)
def test_invalid_split_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        walk_forward_splits(100, **kwargs)


def test_slice_frames_prepends_lookback_bars():
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    window = Window(50, 60)
    assert len(slice_frames(frames, window)["X"]) == 10
    warm = slice_frames(frames, window, lookback=20)["X"]
    assert len(warm) == 30
    assert warm.iloc[0]["close"] == 130.0  # bar 30 close = 100 + 30


def test_slice_frames_clamps_lookback_at_the_start_of_history():
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    out = slice_frames(frames, Window(0, 10), lookback=20)["X"]
    assert len(out) == 10  # cannot look back before bar 0


def test_fold_describe_is_readable():
    frames = linear_bars(START, 100, 100.0, 1.0)
    fold = walk_forward_splits(100, train_size=50, test_size=10)[0]
    text = fold.describe(frames.index)
    assert "fold 0" in text and "train=" in text and "test=" in text
