"""Tracker tests. Every expected value is hand-computed in the comments."""

from datetime import UTC, datetime

import pytest

from kestrel.execution.types import Fill, Side
from kestrel.tracker import Holding, Snapshot, SnapshotLog, summarise_week


def at(day: int, hour: int = 21) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=UTC)


def snap(day: int, cash: float, *holdings: Holding) -> Snapshot:
    return Snapshot(at(day), cash, holdings)


def test_equity_is_cash_plus_marked_holdings():
    # 1_000 + 10*50 + 0.01*40_000 = 1_000 + 500 + 400 = 1_900
    s = snap(1, 1_000, Holding("AAPL", 10, 50), Holding("BTC", 0.01, 40_000))
    assert s.invested == pytest.approx(900)
    assert s.equity == pytest.approx(1_900)


def test_unrealised_pnl_needs_avg_cost():
    assert Holding("X", 4, 30, avg_cost=25).unrealised_pnl == pytest.approx(20)  # (30-25)*4
    assert Holding("X", 4, 30).unrealised_pnl is None


def test_snapshot_rejects_naive_time_and_duplicate_symbols():
    with pytest.raises(ValueError, match="tz-aware"):
        Snapshot(datetime(2026, 9, 1), 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        snap(1, 0, Holding("A", 1, 1), Holding("A", 2, 1))


def test_log_round_trips_and_filters_by_time(tmp_path):
    log = SnapshotLog(tmp_path / "sub" / "snaps.jsonl")
    assert log.load() == []
    a = snap(1, 100, Holding("AAPL", 2, 50, avg_cost=45, venue="trading212"))
    b = snap(3, 120)
    c = snap(5, 130)
    for s in (c, a, b):  # written out of order; load sorts
        log.append(s)
    assert log.load() == [a, b, c]
    assert log.load(start=at(2), end=at(4)) == [b]


def test_weekly_return_without_deposits():
    # start 1_000 cash + 10 @ 100 = 2_000; end 1_000 + 10 @ 110 = 2_100
    # pnl 100, return 100 / 2_000 = 5%
    start, end = snap(1, 1_000, Holding("A", 10, 100)), snap(8, 1_000, Holding("A", 10, 110))
    s = summarise_week([start, end])
    assert s.pnl == pytest.approx(100)
    assert s.return_pct == pytest.approx(0.05)
    assert s.movers[0].change == pytest.approx(0.10)
    assert s.cash_weight == pytest.approx(1_000 / 2_100)


def test_deposits_are_not_counted_as_profit():
    # start 2_000; user pays in 500; end 2_600.
    # pnl = 2_600 - 2_000 - 500 = 100; return = 100 / 2_500 = 4%
    s = summarise_week([snap(1, 2_000), snap(8, 2_600)], net_deposits=500)
    assert s.pnl == pytest.approx(100)
    assert s.return_pct == pytest.approx(0.04)


def test_within_week_drawdown():
    # equity 1_000 -> 1_200 -> 900 -> 1_100; worst = 900/1_200 - 1 = -25%
    snaps = [snap(1, 1_000), snap(2, 1_200), snap(3, 900), snap(4, 1_100)]
    assert summarise_week(snaps).max_drawdown == pytest.approx(-0.25)


def test_fills_counted_only_inside_the_week():
    def fill(day: int, fx: float) -> Fill:
        return Fill("A", Side.BUY, 1, 10, at(day), fx_fee=fx)

    # day-1 fill is at the opening snapshot (already reflected) -> excluded
    # day-9 is after the closing snapshot -> excluded
    fills = [fill(1, 1.0), fill(3, 0.15), fill(8, 0.30), fill(9, 5.0)]
    s = summarise_week([snap(1, 100), snap(8, 100)], fills)
    assert s.trades == 2
    assert s.fees == pytest.approx(0.45)


def test_movers_only_for_symbols_held_at_both_ends_sorted_by_size():
    start = snap(1, 0, Holding("A", 1, 100), Holding("B", 1, 100), Holding("GONE", 1, 5))
    end = snap(8, 0, Holding("A", 1, 103), Holding("B", 1, 90), Holding("NEW", 1, 7))
    s = summarise_week([start, end])
    # |B| = 10% > |A| = 3%; NEW and GONE are absent
    assert [m.symbol for m in s.movers] == ["B", "A"]
    assert s.movers[0].change == pytest.approx(-0.10)
    # holdings sorted by value: NEW 7 < B 90 < A 103
    assert [h.symbol for h in s.holdings] == ["A", "B", "NEW"]


def test_needs_two_snapshots():
    with pytest.raises(ValueError, match="two snapshots"):
        summarise_week([snap(1, 100)])
