"""Technical indicator features computed from OHLCV data."""

from __future__ import annotations

import pandas as pd
import ta


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~25 technical indicator columns to a DataFrame with OHLCV columns.

    Expects columns: open, high, low, close, volume.
    Returns the DataFrame with new feature columns added.
    """
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    # --- Trend ---
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["adx_14"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    df["cci_20"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()

    ichi = ta.trend.IchimokuIndicator(h, l)
    df["ichimoku_a"] = ichi.ichimoku_a()
    df["ichimoku_b"] = ichi.ichimoku_b()

    ema_12 = ta.trend.EMAIndicator(c, window=12).ema_indicator()
    ema_26 = ta.trend.EMAIndicator(c, window=26).ema_indicator()
    df["ema_12"] = ema_12
    df["ema_26"] = ema_26
    df["ema_spread"] = (ema_12 - ema_26) / ema_26

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # --- Momentum ---
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()

    # --- Volume ---
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()
    df["mfi_14"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    return df
