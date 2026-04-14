"""Cross-asset features: BTC correlation, dominance shifts, sector signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_asset_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    product_id: str,
    correlation_window: int = 30,
) -> pd.DataFrame:
    """Add cross-asset features relative to BTC.

    Args:
        df: OHLCV DataFrame for the target coin.
        btc_df: OHLCV DataFrame for BTC-USD (same timeframe/index).
        product_id: e.g. 'ETH-USD'. Skips most features for BTC itself.
        correlation_window: Rolling window for correlation calc.
    """
    if product_id == "BTC-USD":
        df["btc_correlation"] = 1.0
        df["btc_beta"] = 1.0
        df["btc_relative_strength"] = 0.0
        df["btc_return"] = np.log(btc_df["close"] / btc_df["close"].shift(1))
        return df

    # Align indexes
    common = df.index.intersection(btc_df.index)
    if len(common) < correlation_window + 1:
        df["btc_correlation"] = np.nan
        df["btc_beta"] = np.nan
        df["btc_relative_strength"] = np.nan
        df["btc_return"] = np.nan
        return df

    coin_ret = np.log(df["close"] / df["close"].shift(1))
    btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1)).reindex(df.index)

    # Rolling correlation with BTC
    df["btc_correlation"] = coin_ret.rolling(correlation_window).corr(btc_ret)

    # Rolling beta to BTC
    cov = coin_ret.rolling(correlation_window).cov(btc_ret)
    var = btc_ret.rolling(correlation_window).var()
    df["btc_beta"] = cov / var.replace(0, np.nan)

    # Relative strength vs BTC (coin outperforming or underperforming)
    coin_cum = coin_ret.rolling(correlation_window).sum()
    btc_cum = btc_ret.rolling(correlation_window).sum()
    df["btc_relative_strength"] = coin_cum - btc_cum

    df["btc_return"] = btc_ret

    return df


def compute_btc_dominance_change(
    btc_df: pd.DataFrame,
    all_volumes: dict[str, pd.Series],
    window: int = 20,
) -> pd.Series:
    """Estimate BTC dominance change from relative volume shifts.

    Returns a Series of rolling BTC volume share change.
    """
    btc_vol = btc_df["volume"] * btc_df["close"]
    total_vol = btc_vol.copy()
    for vid, vol_series in all_volumes.items():
        if vid != "BTC-USD":
            aligned = vol_series.reindex(btc_vol.index, fill_value=0)
            total_vol = total_vol + aligned

    dominance = btc_vol / total_vol.replace(0, np.nan)
    return dominance.diff(window)
