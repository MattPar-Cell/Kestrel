"""Performance and risk metrics, shared by the backtest and live monitoring.

These are measurements, not decisions — nothing here sizes a position or blocks
a trade. But three conventions are genuinely arguable, so they are explicit
parameters rather than constants buried in the maths:

1. **Annualisation factor.** 252 for daily equity bars, 365 for crypto (which
   trades weekends), 252*6.5 for hourly US equity bars. Get this wrong and every
   Sharpe you report is scaled by a constant you did not intend.
2. **Risk-free rate.** Defaults to 0.0. That is the honest default for a short
   backtest, and it makes Sharpe slightly *flattering* in a high-rate regime —
   at 4% cash rates, a strategy must clear 4% before it has any excess return at
   all. Set it when comparing against a real cash alternative.
3. **Sortino's minimum acceptable return.** Defaults to 0.0, i.e. downside is
   any negative return. Some definitions use the risk-free rate instead; results
   are not comparable across the two, so this is a parameter.

Sharpe on a short sample is noisy. At 250 daily observations the standard error
of an annualised Sharpe is roughly 1/sqrt(years) ≈ 1.0 — so a reported 1.5 is
not distinguishable from 0.5 at any useful confidence. `sharpe_standard_error`
and `summary`'s `sharpe_se` field exist to keep that in view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

#: Trading days per year for daily US/UK equity bars.
TRADING_DAYS_PER_YEAR = 252
#: Calendar days per year, for instruments that trade continuously.
CALENDAR_DAYS_PER_YEAR = 365


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()


def to_returns(equity: pd.Series) -> pd.Series:
    """Simple per-bar returns from an equity curve."""
    return _clean(pd.Series(equity, dtype="float64").pct_change())


def total_return(equity: pd.Series) -> float:
    equity = pd.Series(equity, dtype="float64").dropna()
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by the equity curve's endpoints.

    Endpoint-sensitive by construction — it ignores the path entirely, so a
    strategy that spent the middle 80% of the sample down 50% shows the same
    CAGR as one that went straight up. Read it next to max drawdown, never alone.
    """
    equity = pd.Series(equity, dtype="float64").dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio. `risk_free_rate` is an annual rate.

    Returns 0.0 when volatility is zero — a flat equity curve has no risk-adjusted
    return to speak of, and the alternative (infinity) poisons every aggregate.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    minimum_acceptable_return: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return over *downside* deviation.

    Downside deviation divides by the full observation count, not just the losing
    ones. That is the standard definition: a strategy with few but severe losses
    should not be rewarded for the rarity of its losses.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    mar_per_period = minimum_acceptable_return / periods_per_year
    excess = r - mar_per_period
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt((downside**2).sum() / len(r))
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Running drawdown as a negative fraction of the prior peak."""
    equity = pd.Series(equity, dtype="float64")
    peak = equity.cummax()
    return (equity / peak - 1.0).fillna(0.0)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction (-0.2 == -20%)."""
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of bars spent below a previous peak.

    Often the more decisive number than drawdown depth: a 15% drawdown lasting
    three months is survivable, the same depth lasting two years usually is not.
    """
    dd = drawdown_series(equity)
    longest = current = 0
    for value in dd:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR divided by max drawdown depth."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / mdd


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Gaussian value-at-risk as a positive fraction of capital.

    A 0.02 result at 95% means: on 5% of horizons, expect to lose 2% or more.

    The Gaussian assumption understates tail risk — real return distributions are
    fat-tailed and left-skewed, so this is a floor on the true risk, not an
    estimate of it. `historical_var` and `expected_shortfall` are the sanity
    checks; when they exceed this substantially, believe them, not this.
    """
    from scipy.stats import norm  # local import keeps module import cheap

    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"confidence must be in [0.5, 1.0), got {confidence}")
    z = norm.ppf(1.0 - confidence)
    var = -(r.mean() * horizon + z * r.std(ddof=1) * np.sqrt(horizon))
    return float(max(var, 0.0))


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical VaR: the loss at the (1-confidence) quantile, as a positive number."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return float(max(-np.quantile(r, 1.0 - confidence), 0.0))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Mean loss conditional on breaching VaR. The number that matters in a crisis."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    threshold = np.quantile(r, 1.0 - confidence)
    tail = r[r <= threshold]
    if tail.empty:
        return 0.0
    return float(max(-tail.mean(), 0.0))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Sharpe over a trailing window. Default 63 bars ≈ one quarter of daily bars.

    Use it to see whether an edge is decaying; a headline Sharpe earned entirely
    in the first year of a four-year sample is a dead strategy with good PR.
    """
    r = _clean(returns)
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    excess = r - risk_free_rate / periods_per_year
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std(ddof=1)
    return (mean / sd.replace(0.0, np.nan) * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_sortino(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series:
    r = _clean(returns)
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    return r.rolling(window).apply(
        lambda w: sortino(pd.Series(w), periods_per_year, minimum_acceptable_return), raw=False
    ).rename("rolling_sortino")


def sharpe_standard_error(
    n_observations: int,
    sharpe_ratio: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Approximate standard error of an *annualised* Sharpe estimate.

    Lo (2002), under the iid assumption: the per-period estimate has
    se_p = sqrt((1 + S_p**2 / 2) / n). Annualising multiplies the ratio by
    sqrt(q), so with S_a = S_p * sqrt(q) the annualised error collapses to

        se_a = sqrt((q + S_a**2 / 2) / n)

    With n = years * q that is roughly 1/sqrt(years) for a modest Sharpe — so one
    year of daily bars gives an error bar of about ±1.0 on the reported figure.
    Print this beside any Sharpe from a short sample; it usually makes the point
    on its own.

    Args:
        n_observations: number of return observations (not trades).
        sharpe_ratio: the annualised Sharpe the error is being computed for.
        periods_per_year: must match the annualisation used for `sharpe_ratio`.
    """
    if n_observations < 2:
        return float("inf")
    return float(np.sqrt((periods_per_year + 0.5 * sharpe_ratio**2) / n_observations))


def correlation_matrix(returns: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    """Pairwise correlation of per-symbol returns.

    `min_periods` guards against a correlation computed from a handful of
    overlapping bars, which is noise that the Phase 3 exposure cap would then
    act on as if it were information.
    """
    return pd.DataFrame(returns, dtype="float64").corr(min_periods=min_periods)


def win_rate(trade_pnl: pd.Series | list[float]) -> float:
    """Fraction of trades with strictly positive P&L. Break-even counts as a loss."""
    pnl = _clean(pd.Series(trade_pnl, dtype="float64"))
    if pnl.empty:
        return 0.0
    return float((pnl > 0).sum() / len(pnl))


def profit_factor(trade_pnl: pd.Series | list[float]) -> float:
    """Gross wins divided by gross losses. inf when there are no losing trades."""
    pnl = _clean(pd.Series(trade_pnl, dtype="float64"))
    if pnl.empty:
        return 0.0
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def average_win_loss_ratio(trade_pnl: pd.Series | list[float]) -> float:
    """Mean win size divided by mean loss size. Feeds the Kelly estimate."""
    pnl = _clean(pd.Series(trade_pnl, dtype="float64"))
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    if wins.empty or losses.empty:
        return 0.0
    return float(wins.mean() / -losses.mean())


@dataclass(frozen=True)
class MetricsSummary:
    """One row of results. `as_dict` feeds reports and Telegram messages."""

    total_return: float
    cagr: float
    annualised_volatility: float
    sharpe: float
    sharpe_se: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar: float
    var_95: float
    historical_var_95: float
    expected_shortfall_95: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    fees_paid: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summary(
    equity: pd.Series,
    trade_pnl: pd.Series | list[float] | None = None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    fees_paid: float = 0.0,
) -> MetricsSummary:
    """Compute every headline metric from an equity curve and trade P&Ls."""
    returns = to_returns(equity)
    pnl = pd.Series([] if trade_pnl is None else trade_pnl, dtype="float64")
    s = sharpe(returns, periods_per_year, risk_free_rate)
    return MetricsSummary(
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year),
        annualised_volatility=annualised_volatility(returns, periods_per_year),
        sharpe=s,
        sharpe_se=sharpe_standard_error(len(returns), s, periods_per_year),
        sortino=sortino(returns, periods_per_year),
        max_drawdown=max_drawdown(equity),
        max_drawdown_duration=max_drawdown_duration(equity),
        calmar=calmar(equity, periods_per_year),
        var_95=parametric_var(returns, 0.95, periods_per_year=periods_per_year),
        historical_var_95=historical_var(returns, 0.95),
        expected_shortfall_95=expected_shortfall(returns, 0.95),
        num_trades=len(pnl),
        win_rate=win_rate(pnl),
        profit_factor=profit_factor(pnl),
        avg_win_loss_ratio=average_win_loss_ratio(pnl),
        fees_paid=fees_paid,
    )
