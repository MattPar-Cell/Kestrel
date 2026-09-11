"""Fill model tests. Every price here is computed by hand in the comment."""

from datetime import UTC, datetime

import pytest

from kestrel.backtest.fills import (
    T212_FX_FEE_BPS,
    FillModel,
    FixedBpsSlippage,
    SpreadSlippage,
)
from kestrel.execution.types import OrderIntent, OrderType, Side

TS = datetime(2024, 1, 2, tzinfo=UTC)


def fill(model: FillModel, intent: OrderIntent, **bar):
    defaults = {"bar_open": 100.0, "bar_high": 102.0, "bar_low": 98.0}
    return model.simulate(intent, timestamp=TS, **(defaults | bar))


# ---- slippage --------------------------------------------------------------
def test_fixed_slippage_moves_price_against_the_order():
    s = FixedBpsSlippage(10.0)  # 10bps = 0.1%
    assert s.apply(Side.BUY, 100.0, 102.0, 98.0) == pytest.approx(100.10)
    assert s.apply(Side.SELL, 100.0, 102.0, 98.0) == pytest.approx(99.90)


def test_spread_slippage_scales_with_bar_range():
    s = SpreadSlippage(0.25)  # quarter of the 4.00 range = 1.00
    assert s.apply(Side.BUY, 100.0, 102.0, 98.0) == pytest.approx(101.0)
    assert s.apply(Side.SELL, 100.0, 102.0, 98.0) == pytest.approx(99.0)


def test_negative_slippage_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        FixedBpsSlippage(-1.0)


# ---- market orders ---------------------------------------------------------
def test_market_buy_fills_at_open_plus_slippage():
    m = FillModel(slippage=FixedBpsSlippage(20.0), commission_bps=0.0, fx_fee_bps=0.0)
    f = fill(m, OrderIntent("X", Side.BUY, 10))
    # 100 * (1 + 20/10_000) = 100.20;  notional = 1002.00
    assert f.price == pytest.approx(100.20)
    assert f.notional == pytest.approx(1002.00)
    assert f.total_fees == 0.0
    assert f.cash_delta == pytest.approx(-1002.00)


def test_market_sell_fills_at_open_minus_slippage():
    m = FillModel(slippage=FixedBpsSlippage(20.0), commission_bps=0.0, fx_fee_bps=0.0)
    f = fill(m, OrderIntent("X", Side.SELL, 10), available_quantity=10.0)
    # 100 * (1 - 0.002) = 99.80;  proceeds = 998.00
    assert f.price == pytest.approx(99.80)
    assert f.cash_delta == pytest.approx(+998.00)


# ---- Trading 212 cost structure -------------------------------------------
def test_t212_fx_fee_is_fifteen_bps_and_charged_on_both_sides():
    """A GBP account buying a US stock pays 0.15% each way — ~30bps round trip."""
    m = FillModel(slippage=FixedBpsSlippage(0.0), commission_bps=0.0)
    assert m.fx_fee_bps == T212_FX_FEE_BPS == 15.0

    buy = fill(m, OrderIntent("X", Side.BUY, 10), requires_fx=True)
    # notional 1000.00; fx = 1000 * 15/10_000 = 1.50
    assert buy.commission == 0.0
    assert buy.fx_fee == pytest.approx(1.50)
    assert buy.cash_delta == pytest.approx(-1001.50)

    sell = m.simulate(
        OrderIntent("X", Side.SELL, 10), timestamp=TS, bar_open=100.0, bar_high=102.0,
        bar_low=98.0, requires_fx=True, available_quantity=10.0,
    )
    assert sell.fx_fee == pytest.approx(1.50)
    assert sell.cash_delta == pytest.approx(+998.50)
    # round trip at an unchanged price loses exactly the two FX fees
    assert buy.cash_delta + sell.cash_delta == pytest.approx(-3.00)


def test_no_fx_fee_for_base_currency_instrument():
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    f = fill(m, OrderIntent("X", Side.BUY, 10), requires_fx=False)
    assert f.fx_fee == 0.0


def test_commission_is_zero_by_default_for_t212():
    assert FillModel().commission_bps == 0.0


def test_commission_applies_when_configured():
    m = FillModel(slippage=FixedBpsSlippage(0.0), commission_bps=5.0, fx_fee_bps=0.0)
    f = fill(m, OrderIntent("X", Side.BUY, 10))
    assert f.commission == pytest.approx(0.50)  # 1000 * 5/10_000


# ---- long-only enforcement (Trading 212 Invest / ISA) ---------------------
def test_sell_without_a_position_is_rejected_when_shorting_disallowed():
    m = FillModel(allow_short=False)
    assert fill(m, OrderIntent("X", Side.SELL, 10), available_quantity=0.0) is None
    assert fill(m, OrderIntent("X", Side.SELL, 10), available_quantity=None) is None


def test_oversized_sell_is_clamped_to_the_held_quantity():
    m = FillModel(slippage=FixedBpsSlippage(0.0), allow_short=False)
    f = fill(m, OrderIntent("X", Side.SELL, 10), available_quantity=4.0)
    assert f.quantity == 4.0, "must not short the 6-share excess"


def test_short_is_allowed_when_explicitly_enabled():
    m = FillModel(slippage=FixedBpsSlippage(0.0), allow_short=True)
    f = fill(m, OrderIntent("X", Side.SELL, 10), available_quantity=0.0)
    assert f is not None and f.quantity == 10.0


# ---- minimum order value --------------------------------------------------
def test_order_below_broker_minimum_does_not_fill():
    m = FillModel(slippage=FixedBpsSlippage(0.0), min_order_value=1.0)
    # 0.005 * 100 = 0.50, below the 1.00 minimum
    assert fill(m, OrderIntent("X", Side.BUY, 0.005)) is None
    # 0.02 * 100 = 2.00, above it
    assert fill(m, OrderIntent("X", Side.BUY, 0.02)) is not None


def test_fractional_quantities_are_supported():
    m = FillModel(slippage=FixedBpsSlippage(0.0), min_order_value=0.0)
    f = fill(m, OrderIntent("X", Side.BUY, 0.4567))
    assert f.quantity == pytest.approx(0.4567)
    assert f.notional == pytest.approx(45.67)


# ---- limit orders ---------------------------------------------------------
def test_limit_buy_fills_at_limit_when_bar_trades_through_it():
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    i = OrderIntent("X", Side.BUY, 1, OrderType.LIMIT, limit_price=99.0)
    # open 100 > limit 99, but low 98 <= 99, so it fills at 99 — never better
    f = fill(m, i, bar_open=100.0, bar_high=102.0, bar_low=98.0)
    assert f.price == pytest.approx(99.0)


def test_limit_buy_does_not_fill_when_price_never_reaches_it():
    m = FillModel()
    i = OrderIntent("X", Side.BUY, 1, OrderType.LIMIT, limit_price=95.0)
    assert fill(m, i, bar_open=100.0, bar_high=102.0, bar_low=98.0) is None


def test_limit_buy_gapping_below_the_limit_fills_at_the_open():
    """A favourable gap is real: you get the open, not the limit."""
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    i = OrderIntent("X", Side.BUY, 1, OrderType.LIMIT, limit_price=99.0)
    f = fill(m, i, bar_open=97.0, bar_high=98.0, bar_low=96.0)
    assert f.price == pytest.approx(97.0)


def test_limit_sell_fills_at_limit_when_bar_reaches_up_to_it():
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    i = OrderIntent("X", Side.SELL, 1, OrderType.LIMIT, limit_price=101.0)
    f = fill(m, i, bar_open=100.0, bar_high=102.0, bar_low=98.0, available_quantity=1.0)
    assert f.price == pytest.approx(101.0)


def test_limit_order_without_a_limit_price_is_rejected_at_construction():
    with pytest.raises(ValueError, match="requires limit_price"):
        OrderIntent("X", Side.BUY, 1, OrderType.LIMIT)


# ---- stop orders ----------------------------------------------------------
def test_stop_sell_triggers_and_fills_at_the_stop():
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    i = OrderIntent("X", Side.SELL, 1, OrderType.STOP, stop_price=99.0)
    f = fill(m, i, bar_open=100.0, bar_high=102.0, bar_low=98.0, available_quantity=1.0)
    assert f.price == pytest.approx(99.0)


def test_stop_sell_gapping_below_the_stop_fills_at_the_worse_open():
    """Gap risk is the whole reason a stop is not a guarantee."""
    m = FillModel(slippage=FixedBpsSlippage(0.0))
    i = OrderIntent("X", Side.SELL, 1, OrderType.STOP, stop_price=99.0)
    f = fill(m, i, bar_open=90.0, bar_high=91.0, bar_low=89.0, available_quantity=1.0)
    assert f.price == pytest.approx(90.0), "must not pretend the stop held at 99"


def test_stop_sell_does_not_trigger_above_the_stop():
    m = FillModel()
    i = OrderIntent("X", Side.SELL, 1, OrderType.STOP, stop_price=95.0)
    assert fill(m, i, bar_open=100.0, bar_high=102.0, bar_low=98.0, available_quantity=1.0) is None


def test_stop_order_without_a_stop_price_is_rejected():
    with pytest.raises(ValueError, match="requires stop_price"):
        OrderIntent("X", Side.BUY, 1, OrderType.STOP)


def test_zero_or_negative_quantity_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        OrderIntent("X", Side.BUY, 0)
    with pytest.raises(ValueError, match="must be positive"):
        OrderIntent("X", Side.BUY, -5)


# ---- slippage attribution -------------------------------------------------
def test_slippage_cost_measures_decision_to_fill_gap():
    m = FillModel(slippage=FixedBpsSlippage(0.0), fx_fee_bps=0.0)
    # decided at a close of 99, filled at an open of 100 -> 1.00 worse per share
    f = fill(m, OrderIntent("X", Side.BUY, 10), reference_price=99.0)
    assert f.slippage_cost == pytest.approx(10.0)
