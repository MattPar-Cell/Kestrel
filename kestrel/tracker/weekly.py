"""The weekly summary: what the portfolio did between two snapshots.

Deposits and withdrawals are the trap here. Paying £1,000 into the account
raises equity by £1,000 without earning anything, so

    pnl     = end_equity - start_equity - net_deposits
    return  = pnl / (start_equity + net_deposits)

treating the flow as if it arrived at the start of the week. That is exact when
it did and a slight understatement of the return when it arrived later — fine
for a weekly message, not fine for a track record. Pass `net_deposits` as
money in minus money out; the broker's cash-transaction history has it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from kestrel.execution.types import Fill
from kestrel.risk.metrics import max_drawdown
from kestrel.tracker.snapshots import Holding, Snapshot


@dataclass(frozen=True, slots=True)
class Mover:
    """A symbol held at both ends of the week, and its price change."""

    symbol: str
    start_price: float
    end_price: float

    @property
    def change(self) -> float:
        return self.end_price / self.start_price - 1.0


@dataclass(frozen=True)
class WeeklySummary:
    start: datetime
    end: datetime
    start_equity: float
    end_equity: float
    net_deposits: float
    cash: float
    #: Within-week drawdown of the snapshot equity curve, as a negative fraction.
    #: Deposits mid-week mask drawdowns; this is a rough gauge, not a risk metric.
    max_drawdown: float
    trades: int
    fees: float
    holdings: tuple[Holding, ...]
    #: Biggest absolute price moves first.
    movers: tuple[Mover, ...]

    @property
    def pnl(self) -> float:
        return self.end_equity - self.start_equity - self.net_deposits

    @property
    def return_pct(self) -> float:
        base = self.start_equity + self.net_deposits
        return self.pnl / base if base > 0 else 0.0

    @property
    def cash_weight(self) -> float:
        return self.cash / self.end_equity if self.end_equity > 0 else 0.0

    def weight(self, holding: Holding) -> float:
        return holding.value / self.end_equity if self.end_equity > 0 else 0.0


def summarise_week(
    snapshots: Sequence[Snapshot],
    fills: Sequence[Fill] = (),
    net_deposits: float = 0.0,
) -> WeeklySummary:
    """Summarise from the first to the last snapshot given.

    Fills count toward the week when `start < timestamp <= end` — a fill stamped
    exactly at the opening snapshot is already reflected in it.
    """
    if len(snapshots) < 2:
        raise ValueError("need at least two snapshots (start and end of the week)")
    snaps = sorted(snapshots, key=lambda s: s.timestamp)
    first, last = snaps[0], snaps[-1]

    week_fills = [f for f in fills if first.timestamp < f.timestamp <= last.timestamp]
    equity = pd.Series([s.equity for s in snaps], index=[s.timestamp for s in snaps])

    movers = []
    for h in last.holdings:
        before = first.holding(h.symbol)
        if before is not None and before.price > 0:
            movers.append(Mover(h.symbol, before.price, h.price))
    movers.sort(key=lambda m: abs(m.change), reverse=True)

    return WeeklySummary(
        start=first.timestamp,
        end=last.timestamp,
        start_equity=first.equity,
        end_equity=last.equity,
        net_deposits=net_deposits,
        cash=last.cash,
        max_drawdown=max_drawdown(equity),
        trades=len(week_fills),
        fees=sum(f.total_fees for f in week_fills),
        holdings=tuple(sorted(last.holdings, key=lambda h: h.value, reverse=True)),
        movers=tuple(movers),
    )
