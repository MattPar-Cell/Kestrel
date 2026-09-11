"""Fill simulation.

The single most important rule in this file: **an order decided on bar `t` fills
at the open of bar `t+1`**. Filling at the close of bar `t` would let the
strategy trade at a price it used to make the decision, which inflates results
and is the classic way a backtest lies to you.

Cost model is Trading 212 Invest/ISA shaped:

* **Commission: zero.** Trading 212 charges no per-trade commission on Invest
  and Stocks ISA. `commission_bps` stays configurable so a different broker (or
  a stress test) can be modelled, but 0 is the realistic default here.
* **FX fee: 0.15%** on the converted value whenever the instrument's currency
  differs from the account's. For a GBP account trading US equities this applies
  to *every* trade, both directions, and at ~30bp round-trip it dominates the
  cost model. Getting this wrong is the difference between a viable daily
  strategy and a losing one.
* **Slippage:** a fixed bps haircut against the fill. A constant is a crude
  model; it is honest for liquid large caps on daily bars and optimistic for
  anything thin. `SpreadSlippage` is available when a bar's own range is a
  better guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from kestrel.execution.types import Fill, OrderIntent, OrderType, Side

#: Trading 212's FX conversion fee on Invest/ISA accounts, in basis points.
T212_FX_FEE_BPS = 15.0


class SlippageModel(Protocol):
    def apply(self, side: Side, price: float, high: float, low: float) -> float:
        """Return the effective fill price, moved against the order."""
        ...


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    """Move the fill price against the order by a constant number of bps."""

    bps: float = 2.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError(f"slippage bps must be non-negative, got {self.bps}")

    def apply(self, side: Side, price: float, high: float, low: float) -> float:
        return price * (1.0 + side.sign * self.bps / 10_000.0)


@dataclass(frozen=True, slots=True)
class SpreadSlippage:
    """Charge a fraction of the bar's own high-low range.

    More responsive than a fixed haircut: costs widen in volatile bars, which is
    when they genuinely do widen. Still pessimistic-by-construction only if
    `fraction` is set high enough — it is not a substitute for real tick data.
    """

    fraction: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")

    def apply(self, side: Side, price: float, high: float, low: float) -> float:
        return price + side.sign * self.fraction * max(high - low, 0.0)


@dataclass(frozen=True, slots=True)
class FillModel:
    """Turns an `OrderIntent` plus the next bar into a `Fill` (or nothing).

    Args:
        slippage: how the fill price is degraded.
        commission_bps: per-trade commission. 0 for Trading 212 Invest/ISA.
        fx_fee_bps: conversion fee applied when `requires_fx` is True.
        min_order_value: orders below this notional are dropped unfilled.
            Trading 212 enforces a small minimum (about £1 / $1); sizing can
            produce sub-minimum orders for tiny signals on a small account.
        allow_short: when False, sell orders may only reduce an existing long.
            Trading 212 Invest and Stocks ISA cannot short, so False is correct
            for this broker.
    """

    slippage: SlippageModel = FixedBpsSlippage(2.0)
    commission_bps: float = 0.0
    fx_fee_bps: float = T212_FX_FEE_BPS
    min_order_value: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.fx_fee_bps < 0:
            raise ValueError("fee rates must be non-negative")
        if self.min_order_value < 0:
            raise ValueError("min_order_value must be non-negative")

    def simulate(
        self,
        intent: OrderIntent,
        *,
        timestamp: datetime,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        reference_price: float | None = None,
        requires_fx: bool = False,
        available_quantity: float | None = None,
    ) -> Fill | None:
        """Attempt to fill `intent` against the bar that opens at `timestamp`.

        Returns None when the order cannot fill: a limit price the bar never
        reached, a stop never triggered, a notional below the broker minimum, or
        a short that this account type forbids.

        `available_quantity` is the current long position size; it is used only
        to clamp sells when shorting is disallowed.
        """
        quantity = intent.quantity

        if not self.allow_short and intent.side is Side.SELL:
            if available_quantity is None or available_quantity <= 0:
                return None  # nothing to sell and we cannot go short
            quantity = min(quantity, available_quantity)

        price = self._execution_price(intent, bar_open, bar_high, bar_low)
        if price is None:
            return None

        price = self.slippage.apply(intent.side, price, bar_high, bar_low)
        notional = quantity * price
        if notional < self.min_order_value:
            return None

        return Fill(
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=notional * self.commission_bps / 10_000.0,
            fx_fee=notional * self.fx_fee_bps / 10_000.0 if requires_fx else 0.0,
            reference_price=reference_price,
            reason=intent.reason,
        )

    def _execution_price(
        self, intent: OrderIntent, bar_open: float, bar_high: float, bar_low: float
    ) -> float | None:
        """Pre-slippage price, or None if the order does not execute on this bar.

        Conservative throughout. A limit order that the bar merely *touched*
        fills at the limit price, never better — assuming we got the best price
        inside the bar is the second most common way a backtest flatters itself.
        """
        match intent.order_type:
            case OrderType.MARKET:
                return bar_open

            case OrderType.LIMIT:
                limit = intent.limit_price
                assert limit is not None  # guaranteed by OrderIntent validation
                if intent.side is Side.BUY:
                    if bar_open <= limit:
                        return bar_open  # gapped through in our favour
                    return limit if bar_low <= limit else None
                if bar_open >= limit:
                    return bar_open
                return limit if bar_high >= limit else None

            case OrderType.STOP:
                stop = intent.stop_price
                assert stop is not None
                if intent.side is Side.BUY:
                    if bar_open >= stop:
                        return bar_open
                    return stop if bar_high >= stop else None
                if bar_open <= stop:
                    return bar_open
                return stop if bar_low <= stop else None

            case OrderType.STOP_LIMIT:
                stop, limit = intent.stop_price, intent.limit_price
                assert stop is not None and limit is not None
                triggered = (
                    (bar_high >= stop or bar_open >= stop)
                    if intent.side is Side.BUY
                    else (bar_low <= stop or bar_open <= stop)
                )
                if not triggered:
                    return None
                if intent.side is Side.BUY:
                    return limit if bar_low <= limit else None
                return limit if bar_high >= limit else None

        raise AssertionError(f"unhandled order type {intent.order_type}")
