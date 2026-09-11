"""Order and fill vocabulary shared by the simulated and real brokers.

This module is the seam that lets the backtest and live trading run the same
strategy and risk code. A strategy emits `OrderIntent`s and consumes `Fill`s; it
never knows whether a `Broker` implementation is a simulator or Trading 212.

Nothing here does I/O, so it is safe to import from pure modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Used wherever a signed quantity is needed."""
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A strategy's request to change a position.

    "Intent" rather than "Order" on purpose: risk checks sit between this and
    anything reaching a broker, and they are allowed to shrink or reject it.

    `quantity` is always positive; direction lives in `side`. Fractional
    quantities are permitted — Trading 212 supports fractional shares, and
    volatility-based sizing almost never lands on a whole number.
    """

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    #: Free-text provenance ("ema_cross_up", "drawdown_stop"). Carried into the
    #: trade log and Telegram messages so every fill can be traced to a cause.
    reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"{self.symbol}: quantity must be positive, got {self.quantity}")
        needs_limit = self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
        if needs_limit and self.limit_price is None:
            raise ValueError(f"{self.symbol}: {self.order_type} requires limit_price")
        needs_stop = self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
        if needs_stop and self.stop_price is None:
            raise ValueError(f"{self.symbol}: {self.order_type} requires stop_price")

    @property
    def signed_quantity(self) -> float:
        return self.quantity * self.side.sign


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed (or partially executed) order.

    `price` is the all-in execution price *before* fees; `commission` and
    `fx_fee` are charged separately so reports can attribute cost drag to the
    right source.
    """

    symbol: str
    side: Side
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fx_fee: float = 0.0
    #: Price the decision was based on, for measuring realised slippage.
    reference_price: float | None = None
    reason: str = ""
    order_id: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"{self.symbol}: fill quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"{self.symbol}: fill price must be positive, got {self.price}")
        if self.timestamp.tzinfo is None:
            raise ValueError(f"{self.symbol}: fill timestamp must be tz-aware")

    @property
    def notional(self) -> float:
        """Gross traded value, fees excluded."""
        return self.quantity * self.price

    @property
    def total_fees(self) -> float:
        return self.commission + self.fx_fee

    @property
    def cash_delta(self) -> float:
        """Signed change to cash: buys cost money, sells raise it, fees always cost."""
        return -self.side.sign * self.notional - self.total_fees

    @property
    def slippage_cost(self) -> float:
        """Currency lost to the gap between decision price and fill price.

        Positive means the fill was worse than the reference. Zero if no
        reference price was recorded.
        """
        if self.reference_price is None:
            return 0.0
        return self.side.sign * (self.price - self.reference_price) * self.quantity


@dataclass(frozen=True, slots=True)
class Position:
    """A net holding in one symbol.

    `quantity` is signed: negative means short. Short positions are
    representable here but rejected by the Trading 212 broker adapter, because
    Invest and Stocks ISA accounts are long-only.
    """

    symbol: str
    quantity: float
    avg_price: float

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.quantity


@dataclass
class AccountState:
    """A point-in-time snapshot, shared by the backtest and live monitoring."""

    timestamp: datetime
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0
    fees_paid: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus marked-to-market positions.

        Raises on a missing price rather than treating the position as worthless
        — a silently-zeroed holding would corrupt the equity curve and every
        metric derived from it.
        """
        total = self.cash
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            if symbol not in prices:
                raise KeyError(f"no mark price for open position {symbol!r}")
            total += pos.market_value(prices[symbol])
        return total


@runtime_checkable
class Broker(Protocol):
    """The interface both the simulator and Trading 212 implement."""

    name: str

    async def submit(self, intent: OrderIntent) -> str:
        """Send an order; return a broker-side order id."""
        ...

    async def cancel(self, order_id: str) -> None: ...

    async def account(self) -> AccountState:
        """Current cash, positions, and P&L."""
        ...
