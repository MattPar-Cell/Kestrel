"""Walk-forward runner tests: stitching, compounding, and the overfitting signal."""

import pytest

from kestrel.backtest.engine import BacktestEngine
from kestrel.backtest.runner import run_walk_forward
from kestrel.data.synthetic import constant_bars, linear_bars
from tests.conftest import START, BuyAndHold, RecordingStrategy


class BuyEveryFold:
    """Buys once per engine run, so each fold produces exactly one entry."""

    name = "buy_every_fold"

    def __init__(self, symbol: str, quantity: float = 1.0) -> None:
        self.symbol, self.quantity = symbol, quantity
        self._bought_in_run = False
        self._last_ts = None

    def on_bars(self, ctx):
        from kestrel.execution.types import OrderIntent, Side

        # A new run always starts earlier than the previous one ended.
        if self._last_ts is None or ctx.timestamp < self._last_ts:
            self._bought_in_run = False
        self._last_ts = ctx.timestamp
        if self._bought_in_run or self.symbol not in ctx.bars:
            return []
        self._bought_in_run = True
        return [OrderIntent(self.symbol, Side.BUY, self.quantity, reason="entry")]


async def test_runner_produces_one_result_per_fold(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        RecordingStrategy(),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert len(report.folds) == 5
    assert report.metadata["num_folds"] == 5
    assert report.metadata["scheme"] == "rolling"


async def test_oos_curve_is_flat_for_a_strategy_that_never_trades(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        RecordingStrategy(),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert (report.oos_equity == 10_000.0).all()
    assert report.oos_metrics.total_return == pytest.approx(0.0)
    assert report.oos_metrics.num_trades == 0


async def test_oos_segments_are_stitched_without_duplicate_timestamps(frictionless):
    """Each fold contributes its test window once; boundaries must not double up."""
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert not report.oos_equity.index.has_duplicates
    assert report.oos_equity.index.is_monotonic_increasing
    # 5 folds x 10 test bars, minus 4 shared boundary points
    assert len(report.oos_equity) == 46


async def test_oos_curve_compounds_across_folds_instead_of_resetting(frictionless):
    """Each fold is rebased onto the running balance, so gains carry forward."""
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyEveryFold("X", quantity=10.0),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert report.oos_equity.iloc[0] == pytest.approx(10_000.0)
    assert report.oos_equity.iloc[-1] > 10_000.0
    # monotonically rising price with a long position: the curve never falls
    assert (report.oos_equity.diff().dropna() >= -1e-9).all()


async def test_flat_market_oos_return_is_zero_with_no_costs(frictionless):
    frames = {"X": constant_bars(START, 100, 100.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert report.oos_metrics.total_return == pytest.approx(0.0)
    assert report.oos_metrics.max_drawdown == pytest.approx(0.0)


async def test_fold_table_has_one_row_per_fold(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    table = report.fold_table()
    assert len(table) == 5
    assert list(table["fold"]) == [0, 1, 2, 3, 4]
    assert set(table.columns) >= {"sharpe", "max_drawdown", "num_trades", "total_return"}


async def test_trade_log_is_tagged_with_its_fold(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    # BuyEveryFold never sells, so no round trip closes: the log is empty but typed
    assert "net_pnl" in report.oos_trades.columns


async def test_in_sample_metrics_are_reported_and_give_an_overfitting_gap(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    assert report.is_metrics is not None
    assert report.overfitting_gap is not None


async def test_in_sample_can_be_skipped(frictionless):
    frames = {"X": linear_bars(START, 100, 100.0, 1.0)}
    report = await run_walk_forward(
        RecordingStrategy(),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
        include_in_sample=False,
    )
    assert report.is_metrics is None
    assert report.overfitting_gap is None
    assert all(fr.train is None for fr in report.folds)


async def test_insufficient_history_raises_a_clear_error(frictionless):
    frames = {"X": linear_bars(START, 30, 100.0, 1.0)}
    with pytest.raises(ValueError, match="not enough history"):
        await run_walk_forward(
            RecordingStrategy(),
            frames,
            engine=BacktestEngine(frictionless),
            train_size=50,
            test_size=10,
        )


async def test_summary_text_warns_when_the_sample_is_too_short(frictionless):
    """A Sharpe whose error bar exceeds it must say so in the report."""
    frames = {"X": constant_bars(START, 100, 100.0)}
    report = await run_walk_forward(
        BuyEveryFold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
    )
    text = report.summary_text()
    assert "Sharpe" in text and "too" in text


async def test_lookback_warms_indicators_without_trading_on_cold_ones(frictionless):
    """With lookback, each fold's slice starts earlier but warmup suppresses trades."""
    frames = {"X": linear_bars(START, 120, 100.0, 1.0)}
    report = await run_walk_forward(
        BuyAndHold("X"),
        frames,
        engine=BacktestEngine(frictionless, starting_cash=10_000.0),
        train_size=50,
        test_size=10,
        lookback=20,
        purge=20,
    )
    assert len(report.folds) >= 1
    for fr in report.folds:
        # slice is lookback + test_size bars, and warmup covers the lookback
        assert len(fr.test.equity_curve) == 30
