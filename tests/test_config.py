"""Config tests. Every expected value here is hand-checkable."""

import pytest
from pydantic import ValidationError

from kestrel.config.settings import Environment, Settings, SizingMode


def _settings(**overrides) -> Settings:
    """Build Settings from explicit kwargs only, ignoring any ambient .env."""
    return Settings(_env_file=None, **overrides)


def test_defaults_are_conservative():
    s = _settings()
    assert s.environment is Environment.BACKTEST
    assert s.risk_per_trade == 0.005          # 0.5%, low end of the agreed range
    assert s.sizing_mode is SizingMode.VOL_TARGET
    assert s.kelly_fraction == 0.25           # quarter-Kelly when opted in


def test_drawdown_stop_is_unset_until_chosen():
    s = _settings()
    assert s.max_daily_drawdown is None
    assert s.max_total_drawdown is None
    assert s.drawdown_stop_configured is False
    assert _settings(max_total_drawdown=0.15).drawdown_stop_configured is True


def test_risk_budget_per_trade_is_equity_times_risk():
    # 250_000 * 0.004 == 1_000
    s = _settings(account_equity=250_000, risk_per_trade=0.004)
    assert s.risk_budget_per_trade == pytest.approx(1_000.0)


def test_symbols_parse_from_comma_separated_string():
    s = _settings(symbols="BTC/USDT, ETH/USDT ,SOL/USDT")
    assert s.symbols == ("BTC/USDT", "ETH/USDT", "SOL/USDT")


def test_live_environment_is_rejected():
    with pytest.raises(ValidationError, match="Phase 6"):
        _settings(environment="live")


def test_paper_environment_requires_broker_key():
    with pytest.raises(ValidationError, match="BROKER_API_KEY"):
        _settings(environment="paper")
    ok = _settings(environment="paper", broker_api_key="dummy", broker_api_secret="dummy")
    assert ok.broker_api_key.get_secret_value() == "dummy"


def test_secrets_are_not_printed_in_repr():
    s = _settings(broker_api_secret="super-secret-token")
    assert "super-secret-token" not in repr(s)


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeframe": "2d"},            # not in the allowed set
        {"risk_per_trade": 0.2},        # > 5% cap
        {"risk_per_trade": 0},          # must be positive
        {"account_equity": -1},
        {"max_concurrent_positions": 0},
        {"max_pairwise_correlation": 1.5},
        {"max_total_drawdown": 1.0},    # must be a fraction < 1
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _settings(**overrides)


# ---- Webull specifics ------------------------------------------------------
def test_webull_cost_defaults():
    """US account: no stock commission, 1% crypto spread each side, no FX, USD 5 min."""
    s = _settings()
    assert s.base_currency == "USD"
    assert s.broker_region == "us"
    assert s.commission_bps == 0.0
    assert s.crypto_fee_bps == 100.0
    assert s.fx_fee_bps == 0.0
    assert s.min_order_value == 5.0


def test_broker_region_is_normalised_and_validated():
    assert _settings(broker_region="MY").broker_region == "my"
    with pytest.raises(ValidationError, match="broker_region"):
        _settings(broker_region="moon")


def test_paper_needs_both_halves_of_the_key_pair():
    with pytest.raises(ValidationError, match="BROKER_API_SECRET"):
        _settings(environment="paper", broker_api_key="dummy")


def test_shorting_is_off_by_default():
    """A cash account cannot short, and crypto never can."""
    assert _settings().allow_short is False


def test_base_currency_is_normalised_and_validated():
    assert _settings(base_currency="gbp").base_currency == "GBP"
    with pytest.raises(ValidationError, match="3-letter ISO"):
        _settings(base_currency="POUNDS")


def test_fee_lists_must_be_in_the_universe():
    """A typo here would silently drop a fee from the backtest."""
    ok = _settings(symbols="AAPL,BTC", fx_symbols="AAPL", crypto_symbols="BTC")
    assert ok.fx_symbols == ("AAPL",)
    assert ok.crypto_symbols == ("BTC",)
    with pytest.raises(ValidationError, match="fx_symbols"):
        _settings(symbols="AAPL", fx_symbols="APPL")
    with pytest.raises(ValidationError, match="crypto_symbols"):
        _settings(symbols="AAPL,BTC", crypto_symbols="BTCC")


def test_signal_recipients_parse_and_configured_flag():
    s = _settings(signal_sender="+447700900123", signal_recipients="+447700900456, group.abc=")
    assert s.signal_recipients == ("+447700900456", "group.abc=")
    assert s.signal_configured is True
    assert _settings(signal_sender="+447700900123").signal_configured is False


def test_signal_sender_must_be_international_format():
    with pytest.raises(ValidationError, match="E.164"):
        _settings(signal_sender="07700900123")
