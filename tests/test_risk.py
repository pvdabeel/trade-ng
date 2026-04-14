"""Tests for the 6-layer risk management system."""

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from src.trading.risk import RiskCheckResult, RiskConfig, RiskManager, TradingState


def _make_risk_manager(
    initial_capital: float = 10000,
    max_loss_pct: float = 0.50,
) -> RiskManager:
    config = RiskConfig(max_loss_pct=max_loss_pct)
    client = MagicMock()
    db = MagicMock()
    db.get_capital_record.return_value = None

    rm = RiskManager(config, client, db)
    rm._initial_capital = initial_capital
    rm._capital_floor = initial_capital * (1 - max_loss_pct)
    rm._peak_value = initial_capital
    rm._daily_start_value = initial_capital
    rm._daily_start_time = dt.datetime.now(dt.timezone.utc)
    rm._hourly_start_value = initial_capital
    rm._hourly_start_time = dt.datetime.now(dt.timezone.utc)
    return rm


class TestLayer2CapitalFloor:
    def test_normal_above_floor(self):
        rm = _make_risk_manager(10000, 0.50)
        state = rm._update_state(8000)
        assert state != TradingState.EMERGENCY

    def test_emergency_at_floor(self):
        rm = _make_risk_manager(10000, 0.50)
        state = rm._update_state(5000)
        assert state == TradingState.EMERGENCY

    def test_emergency_below_floor(self):
        rm = _make_risk_manager(10000, 0.50)
        state = rm._update_state(4000)
        assert state == TradingState.EMERGENCY


class TestLayer3ProgressiveThrottling:
    """Isolate drawdown throttling by setting daily/hourly start to the
    test value so the daily-loss-limit layer doesn't fire first."""

    def test_normal_under_10pct(self):
        rm = _make_risk_manager(10000)
        rm._daily_start_value = 9500
        rm._hourly_start_value = 9500
        rm._update_state(9500)
        assert rm.get_position_scale() == 1.0

    def test_throttled_50_at_15pct(self):
        rm = _make_risk_manager(10000)
        rm._daily_start_value = 8500
        rm._hourly_start_value = 8500
        rm._update_state(8500)
        assert rm.get_position_scale() == 0.5

    def test_throttled_25_at_25pct(self):
        rm = _make_risk_manager(10000)
        rm._daily_start_value = 7500
        rm._hourly_start_value = 7500
        rm._update_state(7500)
        assert rm.get_position_scale() == 0.25

    def test_sell_only_at_35pct(self):
        rm = _make_risk_manager(10000)
        rm._daily_start_value = 6500
        rm._hourly_start_value = 6500
        rm._update_state(6500)
        assert rm.state == TradingState.SELL_ONLY
        assert rm.get_position_scale() == 0.1

    def test_peak_tracking(self):
        rm = _make_risk_manager(10000)
        rm._daily_start_value = 12000
        rm._hourly_start_value = 12000
        rm._update_state(12000)
        assert rm._peak_value == 12000
        rm._daily_start_value = 11000
        rm._hourly_start_value = 11000
        rm._update_state(11000)
        assert rm._peak_value == 12000


class TestLayer4PositionSizing:
    def test_position_size_normal(self):
        rm = _make_risk_manager(10000)
        rm._update_state(10000)
        size = rm.calculate_position_size(10000, 0.7, 100, 5000)
        assert size > 0
        assert size <= 10000 * 0.05

    def test_position_size_zero_when_halted(self):
        rm = _make_risk_manager(10000)
        rm._state = TradingState.HALTED_DAILY
        size = rm.calculate_position_size(10000, 0.7, 100, 5000)
        assert size == 0.0

    def test_position_size_scales_with_throttle(self):
        rm1 = _make_risk_manager(10000)
        rm1._update_state(10000)
        size_normal = rm1.calculate_position_size(10000, 0.7, 100, 5000)

        rm2 = _make_risk_manager(10000)
        rm2._update_state(8500)
        size_throttled = rm2.calculate_position_size(10000, 0.7, 100, 5000)

        assert size_throttled < size_normal


class TestLayer5PreTradeValidation:
    def test_buy_allowed_normal(self):
        rm = _make_risk_manager(10000)
        rm._update_state(10000)
        rm.client.get_usd_balance.return_value = 5000
        result = rm.validate_trade(
            "BTC-USD", "BUY", 500, 10000, [], ["BTC-USD"]
        )
        assert result.allowed

    def test_buy_rejected_sell_only(self):
        rm = _make_risk_manager(10000)
        rm._state = TradingState.SELL_ONLY
        result = rm.validate_trade(
            "BTC-USD", "BUY", 500, 10000, [], ["BTC-USD"]
        )
        assert not result.allowed
        assert "Sell-only" in result.reason

    def test_sell_allowed_in_sell_only(self):
        rm = _make_risk_manager(10000)
        rm._state = TradingState.SELL_ONLY
        result = rm.validate_trade(
            "BTC-USD", "SELL", 500, 10000, [], ["BTC-USD"]
        )
        assert result.allowed

    def test_buy_rejected_max_positions(self):
        rm = _make_risk_manager(10000)
        rm._update_state(10000)
        rm.client.get_usd_balance.return_value = 5000
        positions = [{"is_open": True, "product_id": f"COIN{i}-USD"} for i in range(10)]
        result = rm.validate_trade(
            "BTC-USD", "BUY", 500, 10000, positions, ["BTC-USD"]
        )
        assert not result.allowed
        assert "Max open positions" in result.reason

    def test_buy_rejected_unknown_coin(self):
        rm = _make_risk_manager(10000)
        rm._update_state(10000)
        result = rm.validate_trade(
            "SCAM-USD", "BUY", 500, 10000, [], ["BTC-USD", "ETH-USD"]
        )
        assert not result.allowed
        assert "not in approved" in result.reason

    def test_buy_rejected_halted(self):
        rm = _make_risk_manager(10000)
        rm._state = TradingState.HALTED_DAILY
        result = rm.validate_trade(
            "BTC-USD", "BUY", 500, 10000, [], ["BTC-USD"]
        )
        assert not result.allowed
        assert "halted" in result.reason.lower()


class TestStopLoss:
    def test_stop_loss_calculation(self):
        rm = _make_risk_manager()
        sl = rm.calculate_stop_loss(100, 5)
        assert sl == 90  # 100 - 5*2

    def test_take_profit_calculation(self):
        rm = _make_risk_manager()
        tp = rm.calculate_take_profit(100, 5)
        assert tp == 115  # 100 + 5*3

    def test_trailing_stop(self):
        rm = _make_risk_manager()
        new_stop = rm.update_trailing_stop(110, 100, 5)
        assert new_stop == 100  # 110 - 5*2

    def test_check_stop_triggered(self):
        rm = _make_risk_manager()
        assert rm.check_stop_loss(89, 90) is True
        assert rm.check_stop_loss(91, 90) is False
