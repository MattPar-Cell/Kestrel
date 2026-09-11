"""Portfolio accounting tests with fully hand-computed P&L."""

from datetime import UTC, datetime

import pytest

from kestrel.backtest.portfolio import Portfolio
from kestrel.execution.types import Fill, Side


def ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=UTC)


def buy(qty, price, day, fees=0.0, reason=""):
    return Fill("X", Side.BUY, qty, price, ts(day), commission=fees, reason=reason)


def sell(qty, price, day, fees=0.0, reason=""):
    return Fill("X", Side.SELL, qty, price, ts(day), commission=fees, reason=reason)


def test_starting_state():
    p = Portfolio(10_000.0)
    assert p.cash == 10_000.0
    assert p.quantity("X") == 0.0
    assert p.equity({}) == 10_000.0
    assert p.open_symbols == []


def test_non_positive_starting_cash_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        Portfolio(0.0)


def test_single_buy_moves_cash_and_opens_a_position():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    assert p.cash == 9_000.0           # 10_000 - 10*100
    assert p.quantity("X") == 10.0
    assert p.average_price("X") == 100.0
    assert p.equity({"X": 100.0}) == 10_000.0   # 9_000 + 10*100
    assert p.equity({"X": 110.0}) == 10_100.0   # unrealised +100
    assert p.closed_trades == []


def test_equity_raises_on_a_missing_mark_price():
    """A silently-zeroed position would corrupt the whole equity curve."""
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    with pytest.raises(KeyError, match="no mark price"):
        p.equity({})


def test_average_price_is_quantity_weighted():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(buy(30, 110.0, 2))
    # (10*100 + 30*110) / 40 = (1000 + 3300)/40 = 107.50
    assert p.average_price("X") == pytest.approx(107.50)
    assert p.quantity("X") == 40.0


def test_full_round_trip_realises_pnl_and_returns_to_flat():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(10, 120.0, 5))
    # gross = (120 - 100) * 10 = 200
    assert p.realised_pnl == pytest.approx(200.0)
    assert p.cash == pytest.approx(10_200.0)
    assert p.quantity("X") == 0.0
    assert p.open_symbols == []
    assert len(p.closed_trades) == 1
    t = p.closed_trades[0]
    assert t.gross_pnl == pytest.approx(200.0)
    assert t.net_pnl == pytest.approx(200.0)
    assert t.return_pct == pytest.approx(0.20)      # 200 / (100*10)
    assert t.holding_period.days == 4
    assert t.is_win is True


def test_fifo_matching_closes_the_oldest_lot_first():
    """Buy 10@100, buy 10@110, sell 15@120.

    FIFO closes all of the 100 lot (+200) and half of the 110 lot (+50).
    Average-cost would instead report a single trade of 15 @ 105 (+225) — same
    total, different per-trade numbers, and per-trade numbers drive win rate.
    """
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(buy(10, 110.0, 2))
    p.apply(sell(15, 120.0, 3))

    assert len(p.closed_trades) == 2
    first, second = p.closed_trades
    assert (first.quantity, first.entry_price) == (10.0, 100.0)
    assert first.gross_pnl == pytest.approx(200.0)
    assert (second.quantity, second.entry_price) == (5.0, 110.0)
    assert second.gross_pnl == pytest.approx(50.0)
    assert p.realised_pnl == pytest.approx(250.0)

    # 5 shares of the 110 lot remain
    assert p.quantity("X") == 5.0
    assert p.average_price("X") == 110.0
    # cash: 10_000 - 1_000 - 1_100 + 1_800 = 9_700
    assert p.cash == pytest.approx(9_700.0)


def test_partial_close_keeps_the_remainder_open():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(4, 105.0, 2))
    assert p.quantity("X") == 6.0
    assert p.realised_pnl == pytest.approx(20.0)    # (105-100)*4
    assert len(p.closed_trades) == 1


def test_fees_are_charged_to_cash_and_attributed_to_the_trade():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1, fees=2.0))
    p.apply(sell(10, 120.0, 2, fees=3.0))
    assert p.fees_paid == pytest.approx(5.0)
    t = p.closed_trades[0]
    assert t.fees == pytest.approx(5.0)
    assert t.gross_pnl == pytest.approx(200.0)
    assert t.net_pnl == pytest.approx(195.0)        # net of its own costs
    assert p.realised_pnl == pytest.approx(195.0)
    # cash: 10_000 - 1_000 - 2 + 1_200 - 3 = 10_195
    assert p.cash == pytest.approx(10_195.0)


def test_exit_fees_are_split_across_the_lots_one_fill_closes():
    """A sell closing two lots must not dump its whole fee on the first one."""
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(buy(10, 100.0, 2))
    p.apply(sell(20, 110.0, 3, fees=4.0))
    first, second = p.closed_trades
    assert first.fees == pytest.approx(2.0)
    assert second.fees == pytest.approx(2.0)
    assert p.realised_pnl == pytest.approx(196.0)   # 200 gross - 4 fees


def test_breakeven_trade_counts_as_a_loss():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(10, 100.0, 2))
    assert p.closed_trades[0].net_pnl == 0.0
    assert p.closed_trades[0].is_win is False


def test_losing_trade_signs_correctly():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(10, 90.0, 2))
    assert p.realised_pnl == pytest.approx(-100.0)
    assert p.cash == pytest.approx(9_900.0)
    assert p.closed_trades[0].is_win is False


def test_short_round_trip_profits_when_price_falls():
    """Not reachable through the Trading 212 adapter, but the maths must be right."""
    p = Portfolio(10_000.0)
    p.apply(sell(10, 100.0, 1))
    assert p.quantity("X") == -10.0
    assert p.cash == pytest.approx(11_000.0)
    assert p.equity({"X": 100.0}) == pytest.approx(10_000.0)
    p.apply(buy(10, 90.0, 2))
    assert p.realised_pnl == pytest.approx(100.0)   # (90-100)*10*(-1)
    assert p.cash == pytest.approx(10_100.0)


def test_flip_from_long_to_short_closes_then_reopens():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(15, 110.0, 2))
    assert p.realised_pnl == pytest.approx(100.0)   # closed 10 @ +10
    assert p.quantity("X") == -5.0                  # 5 short remains
    assert p.average_price("X") == 110.0
    assert len(p.closed_trades) == 1


def test_trade_log_has_one_row_per_round_trip():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1, reason="entry"))
    p.apply(buy(10, 110.0, 2, reason="add"))
    p.apply(sell(15, 120.0, 3, reason="exit"))
    log = p.trade_log()
    assert len(log) == 2
    assert list(log["net_pnl"]) == pytest.approx([200.0, 50.0])
    assert set(log["exit_reason"]) == {"exit"}
    assert set(log["entry_reason"]) == {"entry"}


def test_fill_log_records_every_fill_including_reducing_ones():
    p = Portfolio(10_000.0)
    p.apply(buy(10, 100.0, 1))
    p.apply(sell(4, 105.0, 2))
    assert len(p.fill_log()) == 2
    assert len(p.trade_log()) == 1


def test_empty_logs_still_have_the_right_columns():
    p = Portfolio(10_000.0)
    assert "net_pnl" in p.trade_log().columns
    assert "slippage_cost" in p.fill_log().columns


def test_multi_symbol_positions_are_independent():
    p = Portfolio(10_000.0)
    p.apply(Fill("A", Side.BUY, 10, 100.0, ts(1)))
    p.apply(Fill("B", Side.BUY, 5, 200.0, ts(1)))
    assert p.quantity("A") == 10.0
    assert p.quantity("B") == 5.0
    assert sorted(p.open_symbols) == ["A", "B"]
    # cash 10_000 - 1_000 - 1_000 = 8_000; equity 8_000 + 1_100 + 1_050 = 10_150
    assert p.cash == pytest.approx(8_000.0)
    assert p.equity({"A": 110.0, "B": 210.0}) == pytest.approx(10_150.0)
