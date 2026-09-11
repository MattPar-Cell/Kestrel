"""Event-driven backtesting harness: fills, walk-forward splits, reports."""

from kestrel.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Context,
    RiskGate,
    Strategy,
)
from kestrel.backtest.fills import (
    T212_FX_FEE_BPS,
    FillModel,
    FixedBpsSlippage,
    SpreadSlippage,
)
from kestrel.backtest.portfolio import ClosedTrade, Portfolio
from kestrel.backtest.runner import FoldResult, WalkForwardReport, run_walk_forward
from kestrel.backtest.walkforward import (
    Fold,
    SplitScheme,
    Window,
    slice_frames,
    walk_forward_splits,
)

__all__ = [
    "T212_FX_FEE_BPS",
    "BacktestEngine",
    "BacktestResult",
    "ClosedTrade",
    "Context",
    "FillModel",
    "FixedBpsSlippage",
    "Fold",
    "FoldResult",
    "Portfolio",
    "RiskGate",
    "SpreadSlippage",
    "SplitScheme",
    "Strategy",
    "WalkForwardReport",
    "Window",
    "run_walk_forward",
    "slice_frames",
    "walk_forward_splits",
]
