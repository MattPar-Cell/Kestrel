"""Engine tests.

The lookahead tests are the reason this file exists. If they pass, a strategy
cannot trade on information it did not have — not because strategies are written
carefully, but because the engine makes it impossible.
"""


import numpy as np
import pandas as pd
import pytest

from kestrel.backtest.engine import BacktestEngine, Context
from kestrel.backtest.fills import FillModel, FixedBpsSlippage
from kestrel.data.synthetic import constant_bars, linear_bars, step_bars
from kestrel.execution.types import OrderIntent, Side
from tests.conftest import START, BuyAndHold, RecordingStrategy


class CheatingStrategy:
    """Tries to buy the instant it sees a price jump.

    In a lookahead-free engine this is useless: by the time the jump appears in a
    closed bar, the next open has already repriced, so the jump is uncapturable.
    """

    name = "cheater"

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.bought = False

    def on_bars(self, ctx: Context) -> list[OrderIntent]:
        bar = ctx.bars.get(self.symbol)
        if self.bought or bar is None or bar.close <= 100.0:
            return []
        self.bought = True
        return [OrderIntent(self.symbol, Side.BUY, 1.0, reason="saw_the_jump")]


class AlwaysBuy:
    name = "always_buy"

    def __init__(self, symbol: str, quantity: float = 1.0) -> None:
        self.symbol, self.quantity = symbol, quantity

    def on_bars(self, ctx: Context) -> list[OrderIntent]:
        if self.symbol not in ctx.bars:
            return []
        return [OrderIntent(self.symbol, Side.BUY, self.quantity, reason="always")]


# ---- the fill-timing contract ---------------------------------------------
async def test_order_fills_at_the_next_bar_open_not_the_decision_close(frictionless):
    """Bars: close 100,101,102,103,104 / open 100,100,101,102,103.

    Buy decided at bar 0's close (100) must fill at bar 1's open, which is 100.
    Equity then tracks 9_900 cash + 1 share:

        bar 0: 10_000 (cash only, nothing filled yet)
        bar 1:  9_900 + 101 = 10_001
        bar 2:  9_900 + 102 = 10_002
        bar 3:  9_900 + 103 = 10_003
        bar 4:  9_900 + 104 = 10_004
    """
    frames = {"X": linear_bars(START, 5, 100.0, 1.0)}
    engine = BacktestEngine(frictionless, starting_cash=10_000.0)
    result = await engine.run(BuyAndHold("X", 1.0), frames)

    assert list(result.equity_curve) == pytest.approx([10_000, 10_001, 10_002, 10_003, 10_004])
    assert result.final_equity == pytest.approx(10_004.0)
    assert len(result.fill_log) == 1
    assert result.fill_log.iloc[0]["price"] == pytest.approx(100.0)
    assert result.fill_log.iloc[0]["timestamp"] == pd.Timestamp("2024-01-02", tz="UTC")


async def test_cheating_strategy_cannot_capture_the_jump_it_saw(frictionless):
    """Bars: close 100,100,100,150,150 / open 100,100,100,100,150.

    The strategy first sees close > 100 at bar 3. Its order fills at bar 4's
    open — which is 150. It buys the top and captures nothing.

    If this test ever reports a profit, the engine has started filling at the
    decision bar's price and every backtest result is invalid.
    """
    frames = {"X": step_bars(START, 5, price=100.0, jump_at=3, jump_to=150.0)}
    engine = BacktestEngine(frictionless, starting_cash=10_000.0)
    result = await engine.run(CheatingStrategy("X"), frames)

    assert len(result.fill_log) == 1
    assert result.fill_log.iloc[0]["price"] == pytest.approx(150.0), "filled at the post-jump open"
    # cash 10_000 - 150 = 9_850, plus one share marked at 150 = 10_000. Zero gain.
    assert result.final_equity == pytest.approx(10_000.0)
    assert result.total_return == pytest.approx(0.0)


async def test_strategy_history_never_contains_a_future_bar():
    """Structural check: the last row of history is always the decision bar."""
    frames = {"X": linear_bars(START, 10, 100.0, 1.0)}
    strategy = RecordingStrategy()
    engine = BacktestEngine(FillModel(), starting_cash=10_000.0)
    await engine.run(strategy, frames)

    assert len(strategy.seen) == 9  # every bar but the last
    for ctx in strategy.seen:
        history = ctx.history["X"]
        assert history.index[-1] == pd.Timestamp(ctx.timestamp)
        assert (history.index <= pd.Timestamp(ctx.timestamp)).all()
        # the decision bar's own close is visible; nothing after it is
        assert history.iloc[-1]["close"] == ctx.bars["X"].close
        assert len(history) == list(frames["X"].index).index(pd.Timestamp(ctx.timestamp)) + 1


async def test_no_decision_is_taken_on_the_final_bar():
    """Nothing can fill after the last bar, so asking is misleading."""
    frames = {"X": linear_bars(START, 4, 100.0, 1.0)}
    strategy = RecordingStrategy()
    await BacktestEngine(FillModel()).run(strategy, frames)
    assert [c.timestamp for c in strategy.seen][-1] == pd.Timestamp("2024-01-03", tz="UTC")


# ---- sanity properties ----------------------------------------------------
async def test_flat_market_with_no_costs_produces_exactly_zero_pnl(frictionless):
    frames = {"X": constant_bars(START, 20, 100.0)}
    result = await BacktestEngine(frictionless, starting_cash=10_000.0).run(
        AlwaysBuy("X", 0.1), frames
    )
    assert result.final_equity == pytest.approx(10_000.0)
    assert (result.equity_curve == 10_000.0).all()


async def test_flat_market_with_costs_loses_exactly_the_fees():
    """A flat market is the cleanest possible test of the cost model.

    20 bars, decisions on bars 0..18 = 19 orders, each 0.1 share at 100 = 10.00
    notional. Slippage 0 and FX 15bps gives 10.00 * 0.0015 = 0.015 per order,
    so total fees = 19 * 0.015 = 0.285.
    """
    frames = {"X": constant_bars(START, 20, 100.0)}
    model = FillModel(
        slippage=FixedBpsSlippage(0.0), commission_bps=0.0, fx_fee_bps=15.0, min_order_value=0.0
    )
    engine = BacktestEngine(model, starting_cash=10_000.0, fx_symbols=frozenset({"X"}))
    result = await engine.run(AlwaysBuy("X", 0.1), frames)

    assert len(result.fill_log) == 19
    assert result.fees_paid == pytest.approx(0.285)
    assert result.final_equity == pytest.approx(10_000.0 - 0.285)


async def test_strategy_that_never_trades_holds_starting_cash_flat():
    frames = {"X": linear_bars(START, 10, 100.0, 5.0)}
    result = await BacktestEngine(FillModel(), starting_cash=10_000.0).run(
        RecordingStrategy(), frames
    )
    assert (result.equity_curve == 10_000.0).all()
    assert result.num_trades == 0
    assert result.fees_paid == 0.0


async def test_warmup_suppresses_trading_but_not_marking(frictionless):
    frames = {"X": linear_bars(START, 10, 100.0, 1.0)}
    engine = BacktestEngine(frictionless, starting_cash=10_000.0)
    result = await engine.run(BuyAndHold("X", 1.0), frames, warmup=5)
    # first decision is at bar 5 (close 105), filling at bar 6's open = 105
    assert result.fill_log.iloc[0]["price"] == pytest.approx(105.0)
    assert len(result.equity_curve) == 10  # still marked on every bar


async def test_warmup_longer_than_the_backtest_is_rejected():
    frames = {"X": linear_bars(START, 5, 100.0, 1.0)}
    with pytest.raises(ValueError, match="warmup"):
        await BacktestEngine(FillModel()).run(BuyAndHold("X"), frames, warmup=5)


async def test_empty_frames_are_rejected():
    with pytest.raises(ValueError, match="no frames"):
        await BacktestEngine(FillModel()).run(BuyAndHold("X"), {})


# ---- cost realism ---------------------------------------------------------
async def test_round_trip_in_a_flat_market_loses_the_fx_round_trip():
    """Buy bar 0, sell bar 2. Price never moves; the loss is 2 x 15bps."""

    class BuyThenSell:
        name = "buy_then_sell"

        def on_bars(self, ctx: Context) -> list[OrderIntent]:
            i = list(ctx.history["X"].index).index(pd.Timestamp(ctx.timestamp))
            if i == 0:
                return [OrderIntent("X", Side.BUY, 10.0, reason="in")]
            if i == 2:
                return [OrderIntent("X", Side.SELL, 10.0, reason="out")]
            return []

    frames = {"X": constant_bars(START, 5, 100.0)}
    model = FillModel(slippage=FixedBpsSlippage(0.0), commission_bps=0.0, fx_fee_bps=15.0)
    engine = BacktestEngine(model, starting_cash=10_000.0, fx_symbols=frozenset({"X"}))
    result = await engine.run(BuyThenSell(), frames)

    # notional 1_000 each way, 1.50 fee each way
    assert result.fees_paid == pytest.approx(3.0)
    assert result.final_equity == pytest.approx(9_997.0)
    assert result.num_trades == 1
    assert result.trade_log.iloc[0]["net_pnl"] == pytest.approx(-3.0)


async def test_long_only_engine_refuses_to_short(frictionless):
    """Trading 212 Invest/ISA cannot short; a naked sell must be rejected."""

    class SellFirst:
        name = "sell_first"

        def on_bars(self, ctx: Context) -> list[OrderIntent]:
            return [OrderIntent("X", Side.SELL, 5.0, reason="naked_short")]

    frames = {"X": constant_bars(START, 5, 100.0)}
    model = FillModel(slippage=FixedBpsSlippage(0.0), allow_short=False, min_order_value=0.0)
    result = await BacktestEngine(model, starting_cash=10_000.0).run(SellFirst(), frames)

    assert len(result.fill_log) == 0
    assert result.rejected_orders == 4
    assert result.final_equity == pytest.approx(10_000.0)


# ---- multi-symbol and gaps ------------------------------------------------
async def test_multi_symbol_equity_is_the_sum_of_both_legs(frictionless):
    """A: 100 -> 104 (+4/share). B: 200 -> 196 (-4/share). One share each.

    Buys fill at bar 1's open: A at 100, B at 200. Cash 10_000 - 300 = 9_700.
    Final mark: 9_700 + 104 + 196 = 10_000 — the two legs cancel exactly.
    """
    frames = {
        "A": linear_bars(START, 5, 100.0, 1.0),
        "B": linear_bars(START, 5, 200.0, -1.0),
    }

    class BuyBoth:
        name = "buy_both"

        def __init__(self) -> None:
            self.done = False

        def on_bars(self, ctx: Context) -> list[OrderIntent]:
            if self.done:
                return []
            self.done = True
            return [OrderIntent(s, Side.BUY, 1.0, reason="entry") for s in ("A", "B")]

    result = await BacktestEngine(frictionless, starting_cash=10_000.0).run(BuyBoth(), frames)
    assert result.final_equity == pytest.approx(10_000.0)
    assert set(result.symbols) == {"A", "B"}
    assert len(result.fill_log) == 2


async def test_symbol_with_no_bar_at_a_timestamp_is_skipped_not_mispriced(frictionless):
    """A missing bar must not be treated as a zero price."""
    a = linear_bars(START, 6, 100.0, 1.0)
    b = a.copy()
    b.iloc[2] = np.nan  # B does not trade on bar 2
    frames = {"A": a, "B": b}

    strategy = RecordingStrategy()
    result = await BacktestEngine(frictionless, starting_cash=10_000.0).run(strategy, frames)

    bar2 = next(c for c in strategy.seen if c.timestamp == pd.Timestamp("2024-01-03", tz="UTC"))
    assert "A" in bar2.bars and "B" not in bar2.bars
    assert (result.equity_curve == 10_000.0).all()


async def test_result_metadata_records_the_run_shape(frictionless):
    frames = {"X": linear_bars(START, 8, 100.0, 1.0)}
    result = await BacktestEngine(frictionless, starting_cash=10_000.0).run(
        BuyAndHold("X"), frames, warmup=2
    )
    assert result.metadata["bars"] == 8
    assert result.metadata["warmup"] == 2
    assert result.metadata["start"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert result.metadata["open_positions_at_end"] == ["X"]
    assert result.strategy_name == "buy_and_hold"


async def test_returns_property_matches_the_equity_curve(frictionless):
    frames = {"X": linear_bars(START, 5, 100.0, 1.0)}
    result = await BacktestEngine(frictionless, starting_cash=10_000.0).run(
        BuyAndHold("X", 1.0), frames
    )
    # equity 10_000, 10_001, ... -> first return = 1/10_000
    assert result.returns.iloc[0] == pytest.approx(1.0 / 10_000.0)
    assert len(result.returns) == 4
