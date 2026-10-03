"""Message text for Signal. Pure: data in, string out.

A recommendation message has to answer three questions without the reader
opening anything else: **what** to trade, **how much** of the money they have
now, and **when**. "How much" is given both as a currency amount and as a share
of current portfolio value, because the amount alone goes stale the moment the
portfolio moves and the share alone cannot be typed into an order ticket.

"When" is a window, not an instant. Kestrel decides on a bar's close and the
backtest fills at the next bar's open; a recommendation acted on hours after
that window is a different trade from the one that was tested, so each message
says when to stop acting on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from kestrel.execution.types import OrderIntent, Side
from kestrel.tracker.weekly import WeeklySummary

_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}


def format_money(amount: float, currency: str) -> str:
    sign = "-" if amount < 0 else ""
    prefix = _SYMBOLS.get(currency)
    body = f"{abs(amount):,.2f}"
    return f"{sign}{prefix}{body}" if prefix else f"{sign}{body} {currency}"


def _pct(x: float, signed: bool = False) -> str:
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


@dataclass(frozen=True, slots=True)
class Recommendation:
    """An intent the human is asked to place by hand."""

    intent: OrderIntent
    #: Price the size was computed from (last close), in base currency.
    price: float
    #: Earliest sensible time to place the order (normally the next session open).
    act_from: datetime
    #: After this, ignore the message: the trade is no longer the tested one.
    act_by: datetime

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"{self.intent.symbol}: price must be positive, got {self.price}")
        if self.act_from.tzinfo is None or self.act_by.tzinfo is None:
            raise ValueError(f"{self.intent.symbol}: act_from/act_by must be tz-aware")
        if self.act_by <= self.act_from:
            raise ValueError(f"{self.intent.symbol}: act_by must be after act_from")

    @property
    def amount(self) -> float:
        return self.intent.quantity * self.price


def _when(rec: Recommendation, tz: ZoneInfo) -> str:
    a, b = rec.act_from.astimezone(tz), rec.act_by.astimezone(tz)
    if a.date() == b.date():
        return f"{a:%a %d %b} {a:%H:%M}–{b:%H:%M}"
    return f"{a:%a %d %b %H:%M} – {b:%a %d %b %H:%M}"


def format_recommendations(
    recs: Sequence[Recommendation],
    equity: float,
    cash: float,
    currency: str,
    timezone: str = "UTC",
) -> str:
    """One message covering every recommendation from a single decision point."""
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    tz = ZoneInfo(timezone)
    m = lambda x: format_money(x, currency)  # noqa: E731

    if not recs:
        return (
            "Kestrel: no trades today.\n"
            f"Portfolio {m(equity)} · cash {m(cash)} ({_pct(cash / equity)})"
        )

    n = len(recs)
    lines = [
        f"Kestrel: {n} trade{'s' if n != 1 else ''}",
        f"Portfolio {m(equity)} · cash {m(cash)} ({_pct(cash / equity)})",
        "",
    ]
    # Sells first: they raise the cash the buys may need.
    ordered = sorted(recs, key=lambda r: r.intent.side is Side.BUY)
    for i, r in enumerate(ordered, 1):
        it = r.intent
        lines.append(f"{i}. {it.side.upper()} {it.symbol}")
        lines.append(
            f"   {m(r.amount)} ({_pct(r.amount / equity)} of portfolio)"
            f" ≈ {it.quantity:.4g} @ {m(r.price)}"
        )
        lines.append(f"   When: {_when(r, tz)} ({timezone})")
        if it.stop_price is not None:
            lines.append(f"   Stop: {m(it.stop_price)}")
        if it.limit_price is not None:
            lines.append(f"   Limit: {m(it.limit_price)}")
        if it.reason:
            lines.append(f"   Why: {it.reason}")
        lines.append("")

    buys = sum(r.amount for r in recs if r.intent.side is Side.BUY)
    sells = sum(r.amount for r in recs if r.intent.side is Side.SELL)
    if buys > cash + sells:
        lines.append(
            f"⚠ Buys total {m(buys)} but cash plus sells is {m(cash + sells)}."
            " Place the sells first, or scale the buys down."
        )
    lines.append("Skip any trade you see after its window closes.")
    return "\n".join(lines).rstrip()


def format_weekly_summary(
    s: WeeklySummary, currency: str, timezone: str = "UTC", top_movers: int = 3
) -> str:
    tz = ZoneInfo(timezone)
    m = lambda x: format_money(x, currency)  # noqa: E731
    start, end = s.start.astimezone(tz), s.end.astimezone(tz)

    lines = [
        f"Kestrel weekly: {start:%d %b} – {end:%d %b %Y}",
        "",
        f"Portfolio  {m(s.end_equity)}",
        f"Week P&L   {m(s.pnl)} ({_pct(s.return_pct, signed=True)})",
    ]
    if s.net_deposits:
        lines.append(f"Deposits   {m(s.net_deposits)} (excluded from P&L)")
    lines += [
        f"Worst dip  {_pct(s.max_drawdown, signed=True)}",
        f"Trades     {s.trades} · fees {m(s.fees)}",
        f"Cash       {m(s.cash)} ({_pct(s.cash_weight)})",
    ]
    if s.holdings:
        lines += ["", "Holdings"]
        for h in s.holdings:
            row = f"  {h.symbol}  {m(h.value)} ({_pct(s.weight(h))})"
            pnl = h.unrealised_pnl
            if pnl is not None:
                row += f" · open P&L {m(pnl)}"
            lines.append(row)
    if s.movers and top_movers > 0:
        lines += ["", "Biggest moves this week"]
        for mv in s.movers[:top_movers]:
            lines.append(f"  {mv.symbol}  {_pct(mv.change, signed=True)}")
    return "\n".join(lines)
