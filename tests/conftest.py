"""Shared fixtures. Everything here is deterministic — no wall clock, no RNG state."""

from datetime import UTC, datetime

import pytest

from kestrel.backtest.engine import Context
from kestrel.backtest.fills import FillModel, FixedBpsSlippage
from kestrel.execution.types import OrderIntent, Side

START = datetime(2024, 1, 1, tzinfo=UTC)


def utc(day: int, month: int = 1, year: int = 2024) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


@pytest.fixture
def frictionless() -> FillModel:
    """Zero-cost fills, so expected equity curves are exact integers."""
    return FillModel(
        slippage=FixedBpsSlippage(0.0), commission_bps=0.0, fx_fee_bps=0.0, min_order_value=0.0
    )


class BuyAndHold:
    """Buy `quantity` on the first decision bar, then never trade again."""

    name = "buy_and_hold"

    def __init__(self, symbol: str, quantity: float = 1.0) -> None:
        self.symbol = symbol
        self.quantity = quantity
        self.bought = False

    def on_bars(self, ctx: Context) -> list[OrderIntent]:
        if self.bought or self.symbol not in ctx.bars:
            return []
        self.bought = True
        return [OrderIntent(self.symbol, Side.BUY, self.quantity, reason="entry")]


class RecordingStrategy:
    """Trades nothing; records every Context it was given.

    Used to prove the engine never hands a strategy a future bar.
    """

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[Context] = []

    def on_bars(self, ctx: Context) -> list[OrderIntent]:
        self.seen.append(ctx)
        return []
