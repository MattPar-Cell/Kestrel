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
    ok = _settings(environment="paper", broker_api_key="dummy")
    assert ok.broker_api_key.get_secret_value() == "dummy"


def test_secrets_are_not_printed_in_repr():
    s = _settings(telegram_bot_token="super-secret-token")
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
