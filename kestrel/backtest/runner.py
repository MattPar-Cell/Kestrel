"""Running a strategy across walk-forward folds and reporting the result.

The headline number is the **concatenated out-of-sample equity curve**: the test
segments of every fold stitched together. In-sample metrics are reported too, but
only as a diagnostic — a large in-sample/out-of-sample gap is the clearest
overfitting signal available, and seeing both side by side is the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from kestrel.backtest.engine import BacktestEngine, BacktestResult, Strategy
from kestrel.backtest.walkforward import Fold, SplitScheme, slice_frames, walk_forward_splits
from kestrel.risk.metrics import TRADING_DAYS_PER_YEAR, MetricsSummary, summary

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: Fold
    train: BacktestResult | None
    test: BacktestResult

    @property
    def index(self) -> int:
        return self.fold.index


@dataclass
class WalkForwardReport:
    """Aggregate of every fold, plus the stitched out-of-sample curve."""

    folds: list[FoldResult]
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    oos_metrics: MetricsSummary
    is_metrics: MetricsSummary | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def overfitting_gap(self) -> float | None:
        """In-sample Sharpe minus out-of-sample Sharpe.

        Treat anything above ~1.0 as a red flag: it means most of the apparent
        edge existed only in the data the parameters were chosen on. A gap near
        zero is not proof of a real edge, but a large gap is near-proof of its
        absence.
        """
        if self.is_metrics is None:
            return None
        return self.is_metrics.sharpe - self.oos_metrics.sharpe

    def fold_table(self) -> pd.DataFrame:
        """One row per fold: out-of-sample metrics, for spotting instability.

        Consistency across folds matters more than the average. A strategy with
        Sharpe 3.0 in one fold and -0.5 in four others has an average that means
        nothing.
        """
        rows = []
        for fr in self.folds:
            m = summary(fr.test.equity_curve, fr.test.trade_log.get("net_pnl"))
            rows.append(
                {
                    "fold": fr.index,
                    "start": fr.test.equity_curve.index[0],
                    "end": fr.test.equity_curve.index[-1],
                    "bars": len(fr.test.equity_curve),
                    "total_return": m.total_return,
                    "sharpe": m.sharpe,
                    "sortino": m.sortino,
                    "max_drawdown": m.max_drawdown,
                    "num_trades": m.num_trades,
                    "win_rate": m.win_rate,
                    "fees": fr.test.fees_paid,
                }
            )
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        m = self.oos_metrics
        lines = [
            f"Out-of-sample over {len(self.folds)} folds, {len(self.oos_equity)} bars",
            f"  total return      {m.total_return:+.2%}",
            f"  CAGR              {m.cagr:+.2%}",
            f"  Sharpe            {m.sharpe:+.2f}  (±{m.sharpe_se:.2f} s.e.)",
            f"  Sortino           {m.sortino:+.2f}",
            f"  max drawdown      {m.max_drawdown:.2%} over {m.max_drawdown_duration} bars",
            f"  trades            {m.num_trades}  (win rate {m.win_rate:.1%})",
            f"  profit factor     {m.profit_factor:.2f}",
            f"  fees paid         {m.fees_paid:,.2f}",
        ]
        gap = self.overfitting_gap
        if gap is not None:
            flag = "  <-- investigate" if gap > 1.0 else ""
            lines.append(
                f"  in-sample Sharpe  {self.is_metrics.sharpe:+.2f}  (gap {gap:+.2f}){flag}"
            )
        if m.sharpe_se > abs(m.sharpe):
            lines.append("  NOTE: Sharpe standard error exceeds the estimate; this sample is too")
            lines.append("        short to distinguish the result from zero.")
        return "\n".join(lines)


async def run_walk_forward(
    strategy: Strategy,
    frames: dict[str, pd.DataFrame],
    *,
    engine: BacktestEngine | None = None,
    train_size: int,
    test_size: int,
    validate_size: int = 0,
    step: int | None = None,
    scheme: SplitScheme | str = SplitScheme.ROLLING,
    purge: int = 0,
    lookback: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    include_in_sample: bool = True,
) -> WalkForwardReport:
    """Run `strategy` over walk-forward folds and aggregate the results.

    Args:
        lookback: indicator warmup bars prepended to each window. Should be at
            least the longest indicator period, or each fold starts trading on a
            cold indicator.
        purge: bars dropped between train and test. Also should be at least the
            longest indicator lookback, to stop test-window features from being
            computed out of training bars.

    Each fold's out-of-sample equity curve is rebased onto the running balance
    before stitching, so the concatenated curve compounds like a single account
    rather than resetting to the starting cash every fold.
    """
    engine = engine or BacktestEngine()
    n_bars = len(next(iter(frames.values())))
    folds = walk_forward_splits(
        n_bars,
        train_size=train_size,
        test_size=test_size,
        validate_size=validate_size,
        step=step,
        scheme=scheme,
        purge=purge,
    )
    if not folds:
        raise ValueError(
            f"not enough history for a single fold: {n_bars} bars, need at least "
            f"{train_size + validate_size + test_size + 2 * purge}"
        )

    results: list[FoldResult] = []
    oos_segments: list[pd.Series] = []
    oos_trade_frames: list[pd.DataFrame] = []
    is_segments: list[pd.Series] = []
    balance = engine.starting_cash

    for fold in folds:
        log.info("%s", fold.describe(next(iter(frames.values())).index))

        test_result = await engine.run(
            strategy, slice_frames(frames, fold.test, lookback=lookback), warmup=lookback
        )
        train_result = None
        if include_in_sample:
            train_result = await engine.run(
                strategy, slice_frames(frames, fold.train, lookback=lookback), warmup=lookback
            )
            is_segments.append(train_result.equity_curve)

        results.append(FoldResult(fold=fold, train=train_result, test=test_result))

        # Rebase this fold's curve onto the running balance, then drop the first
        # point: it is the opening balance, already represented by the previous
        # fold's final point.
        curve = test_result.equity_curve
        rebased = curve / curve.iloc[0] * balance
        oos_segments.append(rebased.iloc[1:] if oos_segments else rebased)
        balance = float(rebased.iloc[-1])

        if not test_result.trade_log.empty:
            tl = test_result.trade_log.copy()
            tl.insert(0, "fold", fold.index)
            oos_trade_frames.append(tl)

    oos_equity = pd.concat(oos_segments)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="first")].sort_index()
    oos_equity.name = "equity"
    oos_trades = (
        pd.concat(oos_trade_frames, ignore_index=True)
        if oos_trade_frames
        else results[0].test.trade_log
    )
    total_fees = sum(fr.test.fees_paid for fr in results)

    return WalkForwardReport(
        folds=results,
        oos_equity=oos_equity,
        oos_trades=oos_trades,
        oos_metrics=summary(
            oos_equity,
            oos_trades.get("net_pnl"),
            periods_per_year=periods_per_year,
            fees_paid=total_fees,
        ),
        is_metrics=(
            summary(
                pd.concat(is_segments).reset_index(drop=True),
                periods_per_year=periods_per_year,
            )
            if is_segments
            else None
        ),
        metadata={
            "scheme": str(SplitScheme(scheme)),
            "train_size": train_size,
            "validate_size": validate_size,
            "test_size": test_size,
            "purge": purge,
            "lookback": lookback,
            "num_folds": len(folds),
            "strategy": getattr(strategy, "name", type(strategy).__name__),
        },
    )
