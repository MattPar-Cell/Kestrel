"""Signal message tests: exact text for the formatters, a mock transport for I/O."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from kestrel.config.settings import Settings
from kestrel.execution.types import OrderIntent, OrderType, Side
from kestrel.notify import (
    Recommendation,
    SignalError,
    SignalNotifier,
    format_money,
    format_recommendations,
    format_weekly_summary,
)
from kestrel.tracker import Holding, Snapshot, summarise_week

# Mon 5 Oct 2026, 13:30-14:30 UTC == 14:30-15:30 BST (UK is UTC+1 in October)
OPEN = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)
LATE = datetime(2026, 10, 5, 14, 30, tzinfo=UTC)


def rec(symbol: str, side: Side, qty: float, price: float, **kw) -> Recommendation:
    return Recommendation(OrderIntent(symbol, side, qty, **kw), price, OPEN, LATE)


def test_format_money():
    assert format_money(1234.5, "GBP") == "£1,234.50"
    assert format_money(-3, "USD") == "-$3.00"
    assert format_money(10, "CHF") == "10.00 CHF"


def test_recommendation_message_says_what_how_much_and_when():
    # 5 shares @ £200 = £1,000 = 10% of £10,000
    r = rec(
        "AAPL",
        Side.BUY,
        5,
        200,
        reason="ema_cross_up",
        order_type=OrderType.STOP,
        stop_price=185,
    )
    text = format_recommendations([r], equity=10_000, cash=2_500, currency="GBP")
    assert text == (
        "Kestrel: 1 trade\n"
        "Portfolio £10,000.00 · cash £2,500.00 (25.0%)\n"
        "\n"
        "1. BUY AAPL\n"
        "   £1,000.00 (10.0% of portfolio) ≈ 5 @ £200.00\n"
        "   When: Mon 05 Oct 14:30–15:30 (Europe/London)\n"
        "   Stop: £185.00\n"
        "   Why: ema_cross_up\n"
        "\n"
        "Skip any trade you see after its window closes."
    )


def test_sells_listed_first_and_overspend_is_flagged():
    # buys 2 * 1_000 = 2_000; cash 500 + sell 0.01 * 50_000 = 1_000 -> warn
    recs = [rec("MSFT", Side.BUY, 2, 1_000), rec("BTC", Side.SELL, 0.01, 50_000)]
    text = format_recommendations(recs, equity=5_000, cash=500, currency="GBP")
    assert text.index("SELL BTC") < text.index("BUY MSFT")
    assert "⚠ Buys total £2,000.00 but cash plus sells is £1,000.00." in text


def test_no_trades_message():
    text = format_recommendations([], equity=1_000, cash=1_000, currency="GBP")
    assert text.startswith("Kestrel: no trades today.")


def test_recommendation_window_must_be_ordered():
    with pytest.raises(ValueError, match="act_by"):
        Recommendation(OrderIntent("A", Side.BUY, 1), 10, LATE, OPEN)


def test_weekly_summary_text():
    start = Snapshot(datetime(2026, 9, 25, 21, tzinfo=UTC), 1_000, (Holding("A", 10, 100),))
    end = Snapshot(
        datetime(2026, 10, 2, 21, tzinfo=UTC), 1_000, (Holding("A", 10, 110, avg_cost=95),)
    )
    text = format_weekly_summary(summarise_week([start, end]), "GBP")
    # equity 2_000 -> 2_100: +£100, +5.0%; A worth 1_100 = 52.4%; open P&L (110-95)*10 = 150
    assert "Kestrel weekly: 25 Sep – 02 Oct 2026" in text
    assert "Week P&L   £100.00 (+5.0%)" in text
    assert "  A  £1,100.00 (52.4%) · open P&L £150.00" in text
    assert "  A  +10.0%" in text
    assert "Deposits" not in text


async def test_signal_send_posts_v2_payload():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"timestamp": "1"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    n = SignalNotifier("http://signal:8080/", "+447700900123", ["+447700900456"], client=client)
    await n.send("hello")
    assert str(seen[0].url) == "http://signal:8080/v2/send"
    assert json.loads(seen[0].content) == {
        "message": "hello",
        "number": "+447700900123",
        "recipients": ["+447700900456"],
    }


async def test_signal_errors_are_raised_not_swallowed():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(400, text="Unregistered user"))
    )
    n = SignalNotifier("http://signal:8080", "+447700900123", ["+447700900456"], client=client)
    with pytest.raises(SignalError, match="400"):
        await n.send("hello")


def test_from_settings_requires_configuration():
    with pytest.raises(ValueError, match="SIGNAL_SENDER"):
        SignalNotifier.from_settings(Settings(_env_file=None))
