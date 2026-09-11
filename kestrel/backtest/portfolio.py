"""Cash, position, and P&L accounting.

Realised P&L uses **FIFO lot matching**: the oldest open lot is closed first.
The alternative (average-cost) gives the same total over a full round trip but
different per-trade numbers, and per-trade numbers are what the win rate and
Kelly estimates are built from — so the choice matters downstream.

Fees are charged to cash when they occur and attributed to the trade that
incurred them, so a trade's reported P&L is net of its own costs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from kestrel.execution.types import AccountState, Fill, Position, Side


@dataclass(slots=True)
class Lot:
    """One open parcel of a position, kept for FIFO matching."""

    quantity: float  # always positive; direction lives on the Trade/Position
    price: float
    timestamp: datetime
    fees: float = 0.0  # entry fees still attributable to this lot


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A completed round trip on one lot (or part of one)."""

    symbol: str
    side: Side  # direction of the *entry*
    quantity: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    fees: float
    entry_reason: str = ""
    exit_reason: str = ""

    @property
    def gross_pnl(self) -> float:
        """P&L before fees. Negative entry side flips the sign correctly."""
        return (self.exit_price - self.entry_price) * self.quantity * self.side.sign

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        """Net return on the capital committed at entry."""
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost else 0.0

    @property
    def holding_period(self) -> pd.Timedelta:
        return pd.Timestamp(self.exit_time) - pd.Timestamp(self.entry_time)

    @property
    def is_win(self) -> bool:
        """Strictly positive net P&L. Break-even counts as a loss, not a win.

        Deliberate: a scratch trade paid costs for nothing, and counting it as a
        win would flatter the win rate that Kelly sizing later depends on.
        """
        return self.net_pnl > 0.0


@dataclass
class Portfolio:
    """Mutable account state driven by fills.

    `starting_cash` is the whole account at t0; there is no margin model, so
    equity is cash plus marked positions and buying power is just cash.
    """

    starting_cash: float
    cash: float = field(init=False)
    realised_pnl: float = field(default=0.0, init=False)
    fees_paid: float = field(default=0.0, init=False)
    _lots: dict[str, deque[Lot]] = field(default_factory=dict, init=False)
    _direction: dict[str, Side] = field(default_factory=dict, init=False)
    _entry_reason: dict[str, str] = field(default_factory=dict, init=False)
    closed_trades: list[ClosedTrade] = field(default_factory=list, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError(f"starting_cash must be positive, got {self.starting_cash}")
        self.cash = float(self.starting_cash)

    # ---- inspection ------------------------------------------------------
    def quantity(self, symbol: str) -> float:
        """Signed net position size."""
        lots = self._lots.get(symbol)
        if not lots:
            return 0.0
        total = sum(lot.quantity for lot in lots)
        return total * self._direction[symbol].sign

    def position(self, symbol: str) -> Position:
        qty = self.quantity(symbol)
        return Position(symbol, qty, self.average_price(symbol) if qty else 0.0)

    def average_price(self, symbol: str) -> float:
        lots = self._lots.get(symbol)
        if not lots:
            return 0.0
        total_qty = sum(lot.quantity for lot in lots)
        return sum(lot.quantity * lot.price for lot in lots) / total_qty

    @property
    def open_symbols(self) -> list[str]:
        return [s for s in self._lots if self.quantity(s) != 0.0]

    def positions(self) -> dict[str, Position]:
        return {s: self.position(s) for s in self.open_symbols}

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for symbol in self.open_symbols:
            if symbol not in prices:
                raise KeyError(f"no mark price for open position {symbol!r}")
            total += self.quantity(symbol) * prices[symbol]
        return total

    def unrealised_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            self.position(s).unrealised_pnl(prices[s]) for s in self.open_symbols if s in prices
        )

    def snapshot(self, timestamp: datetime) -> AccountState:
        return AccountState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions(),
            realised_pnl=self.realised_pnl,
            fees_paid=self.fees_paid,
        )

    # ---- mutation --------------------------------------------------------
    def apply(self, fill: Fill) -> None:
        """Apply a fill: move cash, open/close lots, record any closed trades."""
        self.cash += fill.cash_delta
        self.fees_paid += fill.total_fees
        self.fills.append(fill)

        symbol = fill.symbol
        current = self._direction.get(symbol)
        lots = self._lots.setdefault(symbol, deque())

        if not lots:
            # Opening from flat: this fill sets the position's direction.
            self._direction[symbol] = Side.BUY if fill.side is Side.BUY else Side.SELL
            self._entry_reason[symbol] = fill.reason
            lots.append(Lot(fill.quantity, fill.price, fill.timestamp, fill.total_fees))
            return

        assert current is not None
        if fill.side is current:
            # Adding to the existing position.
            lots.append(Lot(fill.quantity, fill.price, fill.timestamp, fill.total_fees))
            return

        # Reducing, closing, or flipping.
        remaining = self._close_lots(fill, lots, current)
        if remaining > 0:
            # The fill was larger than the position: flip to the other side.
            # The flipped portion carries no entry fee — the whole fee was
            # already attributed to the closing trades above.
            self._direction[symbol] = fill.side
            self._entry_reason[symbol] = fill.reason
            lots.append(Lot(remaining, fill.price, fill.timestamp, 0.0))

    def _close_lots(self, fill: Fill, lots: deque[Lot], entry_side: Side) -> float:
        """Consume lots FIFO against `fill`; return unmatched quantity.

        Exit fees are split across the closed lots in proportion to the quantity
        each one accounts for, so no trade absorbs the whole cost of a fill that
        closed several lots.
        """
        to_close = fill.quantity
        matched = min(to_close, sum(lot.quantity for lot in lots))
        exit_fee_rate = fill.total_fees / fill.quantity if fill.quantity else 0.0

        while to_close > 0 and lots:
            lot = lots[0]
            closed_qty = min(to_close, lot.quantity)
            entry_fee_share = lot.fees * (closed_qty / lot.quantity) if lot.quantity else 0.0
            trade = ClosedTrade(
                symbol=fill.symbol,
                side=entry_side,
                quantity=closed_qty,
                entry_time=lot.timestamp,
                entry_price=lot.price,
                exit_time=fill.timestamp,
                exit_price=fill.price,
                fees=entry_fee_share + exit_fee_rate * closed_qty,
                entry_reason=self._entry_reason.get(fill.symbol, ""),
                exit_reason=fill.reason,
            )
            self.closed_trades.append(trade)
            self.realised_pnl += trade.net_pnl

            lot.fees -= entry_fee_share
            lot.quantity -= closed_qty
            to_close -= closed_qty
            if lot.quantity <= 1e-12:
                lots.popleft()

        if not lots:
            self._direction.pop(fill.symbol, None)
            self._entry_reason.pop(fill.symbol, None)

        return fill.quantity - matched

    # ---- reporting -------------------------------------------------------
    def trade_log(self) -> pd.DataFrame:
        """Closed trades as a frame, one row per round trip."""
        if not self.closed_trades:
            return pd.DataFrame(
                columns=[
                    "symbol", "side", "quantity", "entry_time", "entry_price",
                    "exit_time", "exit_price", "fees", "gross_pnl", "net_pnl",
                    "return_pct", "holding_period", "entry_reason", "exit_reason",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "side": str(t.side),
                    "quantity": t.quantity,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "fees": t.fees,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "holding_period": t.holding_period,
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                }
                for t in self.closed_trades
            ]
        )

    def fill_log(self) -> pd.DataFrame:
        """Every fill, including those that only reduced a position."""
        if not self.fills:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "side", "quantity", "price",
                         "commission", "fx_fee", "slippage_cost", "reason"]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": f.timestamp,
                    "symbol": f.symbol,
                    "side": str(f.side),
                    "quantity": f.quantity,
                    "price": f.price,
                    "commission": f.commission,
                    "fx_fee": f.fx_fee,
                    "slippage_cost": f.slippage_cost,
                    "reason": f.reason,
                }
                for f in self.fills
            ]
        )
