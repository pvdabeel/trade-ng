"""Tests for the order execution engine."""

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from src.data.coinbase_client import OrderResult
from src.models.ensemble import Signal, TradeSignal
from src.trading.executor import OrderExecutor
from src.trading.risk import RiskConfig, RiskManager, TradingState


def _make_executor():
    client = MagicMock()
    db = MagicMock()
    risk_config = RiskConfig()
    risk = RiskManager(risk_config, client, db)
    risk._initial_capital = 10000
    risk._capital_floor = 5000
    risk._peak_value = 10000
    risk._state = TradingState.NORMAL
    risk._daily_start_value = 10000
    risk._daily_start_time = dt.datetime.now(dt.timezone.utc)
    risk._hourly_start_value = 10000
    risk._hourly_start_time = dt.datetime.now(dt.timezone.utc)
    executor = OrderExecutor(client, db, risk, order_timeout_sec=1)
    return executor, client, db, risk


def _make_signal(product_id="BTC-USD", signal=Signal.BUY, strength=0.7):
    return TradeSignal(
        product_id=product_id,
        signal=signal,
        strength=strength,
        xgb_prob=0.7,
        lstm_pred=0.001,
        ensemble_score=0.7,
    )


class TestExecuteSignal:
    def test_hold_signal_returns_none(self):
        executor, _, _, _ = _make_executor()
        sig = _make_signal(signal=Signal.HOLD)
        result = executor.execute_signal(sig, 10000, 50000, 1000, ["BTC-USD"])
        assert result is None

    def test_sell_without_position_returns_none(self):
        executor, _, db, _ = _make_executor()
        db.get_position.return_value = None
        sig = _make_signal(signal=Signal.SELL)
        result = executor.execute_signal(sig, 10000, 50000, 1000, ["BTC-USD"])
        assert result is None

    def test_buy_respects_risk_rejection(self):
        executor, client, db, risk = _make_executor()
        risk._state = TradingState.SELL_ONLY
        client.get_usd_balance.return_value = 5000
        db.get_open_positions.return_value = []
        sig = _make_signal(signal=Signal.BUY)
        result = executor.execute_signal(sig, 10000, 50000, 1000, ["BTC-USD"])
        assert result is None


class TestEmergencyLiquidation:
    def test_liquidate_cancels_and_sells(self):
        executor, client, db, _ = _make_executor()
        client.list_open_orders.return_value = [
            {"order_id": "order1"}, {"order_id": "order2"}
        ]

        pos_mock = MagicMock()
        pos_mock.product_id = "BTC-USD"
        pos_mock.size = 0.1
        db.get_open_positions.return_value = [pos_mock]

        client.place_market_sell.return_value = OrderResult(
            order_id="sell1", product_id="BTC-USD", side="SELL",
            size=0.1, price=None, status="FILLED", raw={}
        )

        executor.emergency_liquidate()

        client.cancel_orders.assert_called_once()
        client.place_market_sell.assert_called_once()
        db.close_position.assert_called_once()


class TestCheckStops:
    def test_stop_loss_triggers_sell(self):
        executor, client, db, risk = _make_executor()

        pos_mock = MagicMock()
        pos_mock.product_id = "ETH-USD"
        pos_mock.entry_price = 3000
        pos_mock.size = 1.0
        pos_mock.stop_loss = 2800
        pos_mock.take_profit = 3500
        pos_mock.highest_price = 3000
        db.get_open_positions.return_value = [pos_mock]

        client.place_market_sell.return_value = OrderResult(
            order_id="sl1", product_id="ETH-USD", side="SELL",
            size=1.0, price=None, status="FILLED", raw={}
        )

        closed = executor.check_stops({"ETH-USD": 2700})
        assert "ETH-USD" in closed

    def test_take_profit_triggers_sell(self):
        executor, client, db, risk = _make_executor()

        pos_mock = MagicMock()
        pos_mock.product_id = "ETH-USD"
        pos_mock.entry_price = 3000
        pos_mock.size = 1.0
        pos_mock.stop_loss = 2800
        pos_mock.take_profit = 3500
        pos_mock.highest_price = 3000
        db.get_open_positions.return_value = [pos_mock]

        client.place_limit_sell.return_value = OrderResult(
            order_id="tp1", product_id="ETH-USD", side="SELL",
            size=1.0, price=3600, status="FILLED", raw={}
        )

        closed = executor.check_stops({"ETH-USD": 3600})
        assert "ETH-USD" in closed

    def test_no_stop_when_price_in_range(self):
        executor, client, db, risk = _make_executor()

        pos_mock = MagicMock()
        pos_mock.product_id = "ETH-USD"
        pos_mock.entry_price = 3000
        pos_mock.size = 1.0
        pos_mock.stop_loss = 2800
        pos_mock.take_profit = 3500
        pos_mock.highest_price = 3000
        db.get_open_positions.return_value = [pos_mock]

        closed = executor.check_stops({"ETH-USD": 3200})
        assert len(closed) == 0
