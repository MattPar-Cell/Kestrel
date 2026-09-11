"""Metrics tests with hand-computed expected values.

Where a closed form exists it is written out in the docstring or comment, so a
future change that breaks the maths fails here rather than quietly reporting a
wrong Sharpe.
"""

import numpy as np
import pandas as pd
import pytest

from kestrel.risk.metrics import (
    TRADING_DAYS_PER_YEAR,
    average_win_loss_ratio,
    cagr,
    calmar,
    correlation_matrix,
    drawdown_series,
    expected_shortfall,
    historical_var,
    max_drawdown,
    max_drawdown_duration,
    parametric_var,
    profit_factor,
    rolling_sharpe,
    sharpe,
    sharpe_standard_error,
    sortino,
    summary,
    to_returns,
    total_return,
    win_rate,
)


def series(values) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


# ---- returns and totals ---------------------------------------------------
def test_to_returns_is_simple_pct_change():
    eq = series([100.0, 110.0, 99.0])
    # 110/100 - 1 = 0.10 ; 99/110 - 1 = -0.10
    assert list(to_returns(eq)) == pytest.approx([0.10, -0.10])


def test_total_return_uses_endpoints():
    assert total_return(series([100.0, 50.0, 120.0])) == pytest.approx(0.20)


def test_cagr_of_doubling_over_one_year_is_one_hundred_percent():
    """253 daily points = 252 periods = exactly 1 year."""
    eq = series(np.linspace(100.0, 200.0, TRADING_DAYS_PER_YEAR + 1))
    assert cagr(eq) == pytest.approx(1.0, abs=1e-9)


def test_cagr_of_quadrupling_over_two_years_is_one_hundred_percent():
    eq = series(np.linspace(100.0, 400.0, 2 * TRADING_DAYS_PER_YEAR + 1))
    assert cagr(eq) == pytest.approx(1.0, abs=1e-9)  # 4 ** (1/2) - 1


def test_degenerate_inputs_return_zero_not_nan():
    for fn in (total_return, cagr, max_drawdown, calmar):
        assert fn(series([100.0])) == 0.0
    assert sharpe(series([])) == 0.0
    assert sortino(series([0.01])) == 0.0


# ---- Sharpe ---------------------------------------------------------------
def test_sharpe_of_constant_returns_is_zero_not_infinity():
    """Zero volatility must not produce inf, which would poison every aggregate."""
    assert sharpe(series([0.01] * 50)) == 0.0


def test_sharpe_hand_computed():
    """Returns alternate +2% / -1%, n = 50.

    mean = 0.005. Every deviation is +/-0.015, so the *population* sd is exactly
    0.015 and the sample sd (ddof=1, which is what pandas and this module use) is
    0.015 * sqrt(50/49) = 0.01515229.

    Sharpe = 0.005 / 0.01515229 * sqrt(252) = (1/3) * sqrt(49/50) * sqrt(252)
           = 5.23832
    """
    r = series([0.02, -0.01] * 25)
    n = len(r)
    assert r.mean() == pytest.approx(0.005)
    assert r.std(ddof=1) == pytest.approx(0.015 * np.sqrt(n / (n - 1)), abs=1e-12)
    expected = (1 / 3) * np.sqrt((n - 1) / n) * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)
    assert sharpe(r) == pytest.approx(5.23832, abs=1e-5)


def test_sharpe_falls_when_a_risk_free_rate_is_charged():
    r = series([0.001] * 100 + [-0.0005] * 100)
    assert sharpe(r, risk_free_rate=0.04) < sharpe(r, risk_free_rate=0.0)


def test_sharpe_annualisation_factor_scales_the_result():
    r = series([0.02, -0.01] * 25)
    assert sharpe(r, 365) == pytest.approx(sharpe(r, 252) * np.sqrt(365 / 252))


def test_sharpe_standard_error_is_about_one_over_root_years():
    # 1 year of daily bars -> se ~ 1.0 ; 4 years -> ~0.5
    assert sharpe_standard_error(252, 1.0, 252) == pytest.approx(1.0, abs=0.01)
    assert sharpe_standard_error(4 * 252, 1.0, 252) == pytest.approx(0.5, abs=0.01)
    assert sharpe_standard_error(1) == float("inf")


# ---- Sortino -------------------------------------------------------------
def test_sortino_exceeds_sharpe_when_downside_is_milder_than_upside():
    r = series([0.05, -0.01] * 25)
    assert sortino(r) > sharpe(r)


def test_sortino_hand_computed_for_one_loss_in_four():
    """Returns +1%, +1%, +1%, -1%.

    mean = 0.005. downside deviation = sqrt(((-0.01)^2) / 4) = 0.005.
    So Sortino = 0.005/0.005 * sqrt(252) = sqrt(252).
    """
    r = series([0.01, 0.01, 0.01, -0.01])
    assert sortino(r) == pytest.approx(np.sqrt(252))


def test_sortino_with_no_downside_returns_zero_not_infinity():
    assert sortino(series([0.01] * 10)) == 0.0


# ---- drawdown -----------------------------------------------------------
def test_max_drawdown_hand_computed():
    """100 -> 120 -> 60 -> 150. Worst decline is 120 to 60 = -50%."""
    assert max_drawdown(series([100.0, 120.0, 60.0, 150.0])) == pytest.approx(-0.50)


def test_max_drawdown_of_a_monotonic_curve_is_zero():
    assert max_drawdown(series([100.0, 101.0, 102.0])) == 0.0


def test_drawdown_series_tracks_distance_below_the_running_peak():
    dd = drawdown_series(series([100.0, 120.0, 90.0, 120.0, 60.0]))
    # peak 100,120,120,120,120 -> 0, 0, -25%, 0, -50%
    assert list(dd) == pytest.approx([0.0, 0.0, -0.25, 0.0, -0.50])


def test_max_drawdown_duration_counts_bars_below_the_peak():
    """100, 90, 95, 99, 101: bars 1-3 are underwater, bar 4 makes a new high."""
    assert max_drawdown_duration(series([100.0, 90.0, 95.0, 99.0, 101.0])) == 3


def test_calmar_is_cagr_over_drawdown_depth():
    """Needs a real drawdown: 100 -> 120 -> 90 (-25%) -> 150 over one year."""
    eq = series(
        np.concatenate(
            [
                np.linspace(100, 120, 84),
                np.linspace(120, 90, 84)[1:],
                np.linspace(90, 150, 86)[1:],
            ]
        )
    )
    assert max_drawdown(eq) == pytest.approx(-0.25)
    assert calmar(eq) == pytest.approx(cagr(eq) / 0.25)


def test_calmar_of_a_drawdown_free_curve_is_zero_not_infinity():
    assert calmar(series(np.linspace(100.0, 200.0, 100))) == 0.0


# ---- VaR ----------------------------------------------------------------
def test_parametric_var_matches_the_gaussian_quantile():
    """Zero-mean returns with sd 2%: 95% VaR = 1.6449 * 0.02 = 3.29%."""
    rng = np.random.default_rng(7)
    r = series(rng.normal(0.0, 0.02, 20_000))
    assert parametric_var(r, 0.95) == pytest.approx(0.0329, abs=0.001)


def test_parametric_var_is_never_negative():
    """A strongly positive drift can imply a 'negative loss'; clamp it at zero."""
    assert parametric_var(series([0.05] * 50 + [0.04] * 50), 0.95) == 0.0


def test_historical_var_and_expected_shortfall_order_correctly():
    rng = np.random.default_rng(11)
    r = series(rng.standard_t(df=3, size=20_000) * 0.01)
    hv = historical_var(r, 0.95)
    es = expected_shortfall(r, 0.95)
    assert es > hv, "shortfall is the mean beyond VaR, so it must be worse"


def test_gaussian_var_understates_a_fat_tailed_distribution():
    """The documented limitation, asserted so it stays true."""
    rng = np.random.default_rng(3)
    r = series(rng.standard_t(df=2.5, size=50_000) * 0.01)
    assert historical_var(r, 0.99) > parametric_var(r, 0.99)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        parametric_var(series([0.01, -0.01] * 10), confidence=0.2)


# ---- rolling ------------------------------------------------------------
def test_rolling_sharpe_has_the_right_shape_and_warmup():
    r = series([0.01, -0.005] * 60)
    rs = rolling_sharpe(r, window=20)
    assert len(rs) == len(r)
    assert rs.iloc[:19].isna().all()
    assert rs.iloc[19:].notna().all()


def test_rolling_sharpe_window_must_be_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        rolling_sharpe(series([0.01] * 10), window=1)


def test_rolling_sharpe_of_a_full_window_matches_the_static_sharpe():
    r = series([0.02, -0.01] * 25)
    assert rolling_sharpe(r, window=len(r)).iloc[-1] == pytest.approx(sharpe(r))


# ---- correlation --------------------------------------------------------
def test_correlation_matrix_identifies_perfect_and_inverse_pairs():
    base = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 10)
    df = pd.DataFrame({"A": base, "B": base * 2.0, "C": -base})
    corr = correlation_matrix(df, min_periods=5)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(-1.0)
    assert corr.loc["A", "A"] == pytest.approx(1.0)


def test_correlation_min_periods_suppresses_thin_estimates():
    df = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, 0.01, -0.02]})
    assert correlation_matrix(df, min_periods=20).isna().all().all()


# ---- trade statistics ---------------------------------------------------
def test_win_rate_treats_breakeven_as_a_loss():
    # 2 wins out of 5 trades
    assert win_rate([10.0, -5.0, 0.0, 20.0, -1.0]) == pytest.approx(0.4)


def test_win_rate_of_no_trades_is_zero():
    assert win_rate([]) == 0.0


def test_profit_factor_hand_computed():
    # gains 30, losses 6 -> 5.0
    assert profit_factor([10.0, 20.0, -5.0, -1.0]) == pytest.approx(5.0)


def test_profit_factor_is_infinite_with_no_losses():
    assert profit_factor([10.0, 20.0]) == float("inf")
    assert profit_factor([]) == 0.0


def test_average_win_loss_ratio_hand_computed():
    # mean win (10+20)/2 = 15 ; mean loss (5+1)/2 = 3 -> ratio 5.0
    assert average_win_loss_ratio([10.0, 20.0, -5.0, -1.0]) == pytest.approx(5.0)


def test_average_win_loss_ratio_needs_both_sides():
    assert average_win_loss_ratio([10.0, 20.0]) == 0.0


# ---- summary ------------------------------------------------------------
def test_summary_is_internally_consistent():
    eq = series([100.0, 110.0, 105.0, 120.0, 115.0, 130.0])
    s = summary(eq, [5.0, -2.0, 8.0], fees_paid=1.25)
    assert s.total_return == pytest.approx(0.30)
    assert s.max_drawdown == pytest.approx(max_drawdown(eq))
    assert s.num_trades == 3
    assert s.win_rate == pytest.approx(2 / 3)
    assert s.fees_paid == 1.25
    assert s.sharpe_se > 0
    assert set(s.as_dict()) >= {"sharpe", "sortino", "max_drawdown", "var_95"}


def test_summary_with_no_trades_does_not_crash():
    s = summary(series([100.0, 101.0, 102.0]))
    assert s.num_trades == 0
    assert s.win_rate == 0.0
