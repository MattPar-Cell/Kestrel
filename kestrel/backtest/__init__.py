"""Event-driven backtesting harness: fills, walk-forward splits, reports."""

from kestrel.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Context,
    RiskGate,
    Strategy,
)
from kestrel.backtest.fills import (
    WEBULL_CRYPTO_FEE_BPS,
    WEBULL_MIN_ORDER_VALUE,
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
    "WEBULL_CRYPTO_FEE_BPS",
    "WEBULL_MIN_ORDER_VALUE",
    "WalkForwardReport",
    "Window",
    "run_walk_forward",
    "slice_frames",
    "walk_forward_splits",
]
