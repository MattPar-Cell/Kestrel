"""Walk-forward splitting.

A single train/test split tells you almost nothing: one lucky test window reads
as a strategy. Walk-forward re-fits repeatedly and concatenates the out-of-sample
segments, so the reported result is the average over many regimes rather than
one draw.

Two schemes, differing in what the training window does as time advances:

* **Rolling** — fixed-length train window that slides forward. Adapts to regime
  change, and each fit sees the same amount of data, so results are comparable
  across folds.
* **Anchored** — train window always starts at t0 and grows. Uses all available
  history, but early folds fit on much less data than late ones, and an ancient
  regime keeps influencing the fit forever.

Rolling is the default. For a daily-bar momentum strategy, a regime from eight
years ago is not evidence about next month.

The `validate` window exists so parameter selection never touches `test`. Pick
parameters on train, choose among them on validate, and report only on test. If
you tune after looking at test numbers, test stops being out-of-sample and the
reported Sharpe becomes fiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SplitScheme(StrEnum):
    ROLLING = "rolling"
    ANCHORED = "anchored"


@dataclass(frozen=True, slots=True)
class Window:
    """Half-open bar-index range [start, stop)."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(f"empty window [{self.start}, {self.stop})")

    def __len__(self) -> int:
        return self.stop - self.start

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.start : self.stop]

    def timestamps(self, index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
        return index[self.start], index[self.stop - 1]


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold. `validate` is None when validation is disabled."""

    index: int
    train: Window
    validate: Window | None
    test: Window

    @property
    def windows(self) -> dict[str, Window | None]:
        return {"train": self.train, "validate": self.validate, "test": self.test}

    def describe(self, index: pd.DatetimeIndex) -> str:
        parts = [f"fold {self.index}"]
        for name, w in self.windows.items():
            if w is None:
                continue
            first, last = w.timestamps(index)
            parts.append(f"{name}={first:%Y-%m-%d}→{last:%Y-%m-%d} ({len(w)})")
        return "  ".join(parts)


def walk_forward_splits(
    n_bars: int,
    *,
    train_size: int,
    test_size: int,
    validate_size: int = 0,
    step: int | None = None,
    scheme: SplitScheme | str = SplitScheme.ROLLING,
    purge: int = 0,
) -> list[Fold]:
    """Generate walk-forward folds over `n_bars` bars.

    Args:
        train_size: training bars per fold (the *initial* size when anchored).
        test_size: out-of-sample bars per fold.
        validate_size: bars between train and test for model selection.
        step: bars to advance per fold. Defaults to `test_size`, which makes the
            test segments contiguous and non-overlapping — overlapping test
            windows double-count the same bars and make results look more
            significant than they are.
        purge: bars dropped between the end of training and the start of the next
            window. Set this to the longest lookback any indicator uses: without
            it, the first test bars are predicted by features computed partly
            from training data, which leaks.

    Returns:
        Folds in chronological order. Empty if there is not enough history.
    """
    scheme = SplitScheme(scheme)
    for name, value in (("train_size", train_size), ("test_size", test_size)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if validate_size < 0 or purge < 0:
        raise ValueError("validate_size and purge must be non-negative")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    folds: list[Fold] = []
    train_start = 0
    fold_index = 0

    while True:
        train_stop = train_start + train_size
        cursor = train_stop + purge

        validate: Window | None = None
        if validate_size:
            validate = Window(cursor, cursor + validate_size)
            cursor = validate.stop + purge

        test_stop = cursor + test_size
        if test_stop > n_bars:
            break

        folds.append(
            Fold(
                index=fold_index,
                train=Window(train_start, train_stop),
                validate=validate,
                test=Window(cursor, test_stop),
            )
        )
        fold_index += 1
        if scheme is SplitScheme.ROLLING:
            train_start += step
        else:
            train_size += step  # anchored: start stays at 0, window grows

    return folds


def slice_frames(
    frames: dict[str, pd.DataFrame], window: Window, *, lookback: int = 0
) -> dict[str, pd.DataFrame]:
    """Slice every symbol to `window`, optionally prepending warmup bars.

    `lookback` extends the slice *backwards* so indicators are warm at the
    window's first bar. Pair it with `BacktestEngine.run(warmup=lookback)` so
    those bars feed the indicators but generate no trades — otherwise the
    strategy trades on a cold EMA at the start of every fold.
    """
    start = max(0, window.start - lookback)
    return {s: df.iloc[start : window.stop] for s, df in frames.items()}
