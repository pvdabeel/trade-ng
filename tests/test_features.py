"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.cross_asset import add_cross_asset_features
from src.features.market import add_market_features
from src.features.technical import add_technical_features


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = low + (high - low) * np.random.rand(n)
    volume = np.abs(np.random.randn(n) * 1000) + 100

    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestTechnicalFeatures:
    def test_adds_rsi(self):
        df = add_technical_features(_make_ohlcv())
        assert "rsi_14" in df.columns
        valid = df["rsi_14"].dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_adds_macd(self):
        df = add_technical_features(_make_ohlcv())
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_adds_bollinger_bands(self):
        df = add_technical_features(_make_ohlcv())
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_adds_atr(self):
        df = add_technical_features(_make_ohlcv())
        assert "atr_14" in df.columns
        valid = df["atr_14"].dropna()
        assert (valid >= 0).all()

    def test_feature_count(self):
        df = _make_ohlcv()
        orig_cols = len(df.columns)
        df = add_technical_features(df)
        new_cols = len(df.columns) - orig_cols
        assert new_cols >= 20


class TestMarketFeatures:
    def test_adds_returns(self):
        df = add_market_features(_make_ohlcv())
        assert "log_return_1" in df.columns
        assert "log_return_5" in df.columns

    def test_adds_volatility(self):
        df = add_market_features(_make_ohlcv(), volatility_windows=[5, 20])
        assert "volatility_5" in df.columns
        assert "volatility_20" in df.columns

    def test_volume_zscore(self):
        df = add_market_features(_make_ohlcv())
        assert "volume_zscore" in df.columns

    def test_cyclical_time(self):
        df = add_market_features(_make_ohlcv())
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        valid = df["hour_sin"].dropna()
        assert valid.min() >= -1
        assert valid.max() <= 1


class TestCrossAssetFeatures:
    def test_btc_self_correlation(self):
        btc = _make_ohlcv()
        df = add_cross_asset_features(btc.copy(), btc, "BTC-USD")
        assert "btc_correlation" in df.columns
        assert (df["btc_correlation"] == 1.0).all()

    def test_other_coin_correlation(self):
        btc = _make_ohlcv(200)
        eth = _make_ohlcv(200)
        df = add_cross_asset_features(eth, btc, "ETH-USD", correlation_window=20)
        assert "btc_correlation" in df.columns
        assert "btc_beta" in df.columns
        assert "btc_relative_strength" in df.columns
