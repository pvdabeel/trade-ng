"""Market microstructure features: returns, volatility, volume profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(
    df: pd.DataFrame,
    volatility_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add market-derived features to an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume.
    """
    if volatility_windows is None:
        volatility_windows = [5, 10, 20, 50]

    c = df["close"]

    # Log returns at multiple horizons
    df["log_return_1"] = np.log(c / c.shift(1))
    df["log_return_5"] = np.log(c / c.shift(5))
    df["log_return_10"] = np.log(c / c.shift(10))

    # Realized volatility (annualised std of log returns over windows)
    for w in volatility_windows:
        df[f"volatility_{w}"] = df["log_return_1"].rolling(w).std() * np.sqrt(365 * 24)

    # Volume z-score (how unusual is current volume)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # Volume change ratio
    df["volume_change"] = df["volume"] / df["volume"].shift(1)

    # High-low range as pct of close (proxy for intrabar volatility)
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Close position within bar range
    bar_range = df["high"] - df["low"]
    df["close_position"] = np.where(
        bar_range > 0,
        (df["close"] - df["low"]) / bar_range,
        0.5,
    )

    # Price momentum (distance from N-period SMA)
    for w in [10, 20, 50]:
        sma = c.rolling(w).mean()
        df[f"price_vs_sma_{w}"] = (c - sma) / sma

    # Hour-of-day cyclical encoding (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df
