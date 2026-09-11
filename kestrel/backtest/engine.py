"""Event-driven backtest engine.

The loop, per bar `i`:

1. **Fill** orders that were decided at the close of bar `i-1`, at bar `i`'s open.
2. **Mark** the portfolio at bar `i`'s close and record one equity point.
3. **Decide**: hand the strategy history up to and including bar `i`'s close; it
   returns intents, which queue for bar `i+1`'s open.

Steps 1 and 3 are separated by a full bar on purpose. A strategy physically
cannot act on information it has not been given, because the only prices it ever
sees are closes at or before its decision point, and the only prices it ever
trades at are opens strictly after it. That structural guarantee is worth more
than any amount of careful coding inside a strategy.

The engine is async and drives `Strategy` through the same method the live
runner will call, so the Phase 6 loop differs only in where bars come from and
which `Broker` receives the intents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from kestrel.backtest.fills import FillModel
from kestrel.backtest.portfolio import Portfolio
from kestrel.data.types import Bar, validate_bars
from kestrel.execution.types import AccountState, OrderIntent, Side

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    """Everything a strategy is allowed to see at one decision point."""

    timestamp: datetime
    #: Bars up to and **including** `timestamp`. Never contains a future row.
    history: dict[str, pd.DataFrame]
    #: The bar that just closed, for symbols that traded at `timestamp`.
    bars: dict[str, Bar]
    account: AccountState
    equity: float

    def close(self, symbol: str) -> float | None:
        bar = self.bars.get(symbol)
        return bar.close if bar else None

    def quantity(self, symbol: str) -> float:
        pos = self.account.positions.get(symbol)
        return pos.quantity if pos else 0.0


class Strategy(Protocol):
    """Implemented once, run by both the backtest and the live loop."""

    name: str

    def on_bars(self, ctx: Context) -> list[OrderIntent]:
        """Return orders to submit at the next bar's open. May be empty."""
        ...


class RiskGate(Protocol):
    """Sits between the strategy and the broker. Phase 3 fills this in.

    May shrink or drop intents (position limits, correlation caps, drawdown
    hard-stop) and may inject its own (forced liquidation).
    """

    def filter(self, intents: list[OrderIntent], ctx: Context) -> list[OrderIntent]: ...


@dataclass
class BacktestResult:
    """Output of one run."""

    equity_curve: pd.Series
    trade_log: pd.DataFrame
    fill_log: pd.DataFrame
    starting_cash: float
    final_equity: float
    fees_paid: float
    rejected_orders: int = 0
    strategy_name: str = ""
    symbols: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        """Simple per-bar returns of the equity curve."""
        return self.equity_curve.pct_change().dropna()

    @property
    def total_return(self) -> float:
        return self.final_equity / self.starting_cash - 1.0

    @property
    def num_trades(self) -> int:
        return len(self.trade_log)


class BacktestEngine:
    def __init__(
        self,
        fill_model: FillModel | None = None,
        *,
        starting_cash: float = 100_000.0,
        risk_gate: RiskGate | None = None,
        fx_symbols: frozenset[str] | None = None,
    ) -> None:
        """
        Args:
            fx_symbols: symbols whose currency differs from the account's, so FX
                fees apply. For a GBP Trading 212 account trading US equities
                this is every symbol — pass them all, or the backtest will
                understate costs by ~15bps per side.
        """
        self.fill_model = fill_model or FillModel()
        self.starting_cash = starting_cash
        self.risk_gate = risk_gate
        self.fx_symbols = fx_symbols or frozenset()

    async def run(
        self,
        strategy: Strategy,
        frames: dict[str, pd.DataFrame],
        *,
        warmup: int = 0,
    ) -> BacktestResult:
        """Run `strategy` over `frames`.

        Args:
            warmup: bars at the start during which the strategy is marked but
                not consulted. Indicators need history; acting on an EMA that
                has seen three bars is noise, not signal.
        """
        if not frames:
            raise ValueError("no frames to backtest")
        for symbol, df in frames.items():
            validate_bars(df.dropna(subset=["close"]), symbol)

        symbols = tuple(frames)
        index = self._shared_index(frames)
        if warmup >= len(index):
            raise ValueError(
                f"warmup ({warmup}) must be shorter than the backtest ({len(index)} bars)"
            )

        portfolio = Portfolio(self.starting_cash)
        pending: list[OrderIntent] = []
        rejected = 0
        equity_points: list[float] = []
        last_price: dict[str, float] = {}

        for i, timestamp in enumerate(index):
            row = {s: frames[s].iloc[i] for s in symbols}
            tradeable = {s: r for s, r in row.items() if not np.isnan(r["close"])}
            for s, r in tradeable.items():
                last_price[s] = float(r["close"])

            # 1. Fill what was decided at the previous close, at this open.
            for intent in pending:
                r = tradeable.get(intent.symbol)
                if r is None:
                    log.debug("%s: no bar at %s, order dropped", intent.symbol, timestamp)
                    rejected += 1
                    continue
                held = portfolio.quantity(intent.symbol)
                fill = self.fill_model.simulate(
                    intent,
                    timestamp=timestamp,
                    bar_open=float(r["open"]),
                    bar_high=float(r["high"]),
                    bar_low=float(r["low"]),
                    reference_price=last_price.get(intent.symbol),
                    requires_fx=intent.symbol in self.fx_symbols,
                    available_quantity=max(held, 0.0) if intent.side is Side.SELL else None,
                )
                if fill is None:
                    rejected += 1
                    continue
                portfolio.apply(fill)
            pending = []

            # 2. Mark to market at this close.
            equity_points.append(portfolio.equity(last_price))

            # 3. Decide, using only closed bars.
            if i < warmup or i == len(index) - 1:
                # No point deciding on the final bar: nothing can fill after it.
                continue

            ctx = Context(
                timestamp=timestamp,
                history={s: frames[s].iloc[: i + 1] for s in symbols},
                bars={s: self._to_bar(s, timestamp, r) for s, r in tradeable.items()},
                account=portfolio.snapshot(timestamp),
                equity=equity_points[-1],
            )
            intents = strategy.on_bars(ctx)
            if self.risk_gate is not None:
                intents = self.risk_gate.filter(intents, ctx)
            pending = list(intents)

        return BacktestResult(
            equity_curve=pd.Series(equity_points, index=index, name="equity"),
            trade_log=portfolio.trade_log(),
            fill_log=portfolio.fill_log(),
            starting_cash=self.starting_cash,
            final_equity=equity_points[-1],
            fees_paid=portfolio.fees_paid,
            rejected_orders=rejected,
            strategy_name=getattr(strategy, "name", type(strategy).__name__),
            symbols=symbols,
            metadata={
                "bars": len(index),
                "warmup": warmup,
                "start": index[0],
                "end": index[-1],
                "open_positions_at_end": portfolio.open_symbols,
            },
        )

    @staticmethod
    def _shared_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Union of all symbols' timestamps, and reindex every frame onto it.

        Mutates nothing: callers pass frames already aligned by
        `data.loader.align_frames`, and this re-checks rather than trusting it.
        """
        indexes = list(frames.values())
        shared = indexes[0].index
        for df in indexes[1:]:
            shared = shared.union(df.index)
        shared = shared.sort_values()
        for symbol, df in frames.items():
            if not df.index.equals(shared):
                frames[symbol] = df.reindex(shared)
        return shared

    @staticmethod
    def _to_bar(symbol: str, timestamp: datetime, row: pd.Series) -> Bar:
        return Bar(
            symbol=symbol,
            timestamp=pd.Timestamp(timestamp).to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]) if not np.isnan(row["volume"]) else 0.0,
        )
