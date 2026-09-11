"""Position sizing, portfolio constraints, and metrics.

Sizing and constraints arrive in Phase 3. `metrics` is here already because both
the backtest and live monitoring report from it.
"""

from kestrel.risk.metrics import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    MetricsSummary,
    correlation_matrix,
    drawdown_series,
    expected_shortfall,
    historical_var,
    max_drawdown,
    max_drawdown_duration,
    parametric_var,
    rolling_sharpe,
    rolling_sortino,
    sharpe,
    sharpe_standard_error,
    sortino,
    summary,
    to_returns,
    win_rate,
)

__all__ = [
    "CALENDAR_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_YEAR",
    "MetricsSummary",
    "correlation_matrix",
    "drawdown_series",
    "expected_shortfall",
    "historical_var",
    "max_drawdown",
    "max_drawdown_duration",
    "parametric_var",
    "rolling_sharpe",
    "rolling_sortino",
    "sharpe",
    "sharpe_standard_error",
    "sortino",
    "summary",
    "to_returns",
    "win_rate",
]
