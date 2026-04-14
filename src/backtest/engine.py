"""Event-driven backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.ensemble import EnsembleCombiner, Signal
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostSignalModel
from src.trading.risk import RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    product_id: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    entry_time: object
    exit_time: object


@dataclass
class BacktestPosition:
    product_id: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    highest_price: float
    entry_idx: int


@dataclass
class BacktestResult:
    initial_capital: float
    equity_curve: list[float]
    trades: list[BacktestTrade]
    timestamps: list[object]


class BacktestEngine:
    """Simulates trading on historical data with realistic constraints."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_config: RiskConfig | None = None,
        xgb_model: XGBoostSignalModel | None = None,
        lstm_model: LSTMForecaster | None = None,
        ensemble: EnsembleCombiner | None = None,
        maker_fee_pct: float = 0.006,
        taker_fee_pct: float = 0.012,
    ):
        self.initial_capital = initial_capital
        self.risk = risk_config or RiskConfig()
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.ensemble = ensemble or EnsembleCombiner()
        self.maker_fee = maker_fee_pct
        self.taker_fee = taker_fee_pct

        self._data: dict[str, pd.DataFrame] = {}

    def add_data(self, product_id: str, df: pd.DataFrame) -> None:
        self._data[product_id] = df

    def _precompute_signals(self) -> dict[str, pd.DataFrame]:
        """Pre-compute all model predictions once per coin to avoid per-step overhead."""
        signal_cache: dict[str, pd.DataFrame] = {}

        if self.xgb is None or self.lstm is None:
            return signal_cache

        for pid, df in self._data.items():
            try:
                xgb_probs = self.xgb.predict(df)
                lstm_preds = self.lstm.predict(df)

                signals_df = pd.DataFrame(
                    {"xgb_prob": xgb_probs, "lstm_pred": lstm_preds},
                    index=df.index,
                )
                signal_cache[pid] = signals_df
                logger.info("Pre-computed signals for %s (%d rows)", pid, len(df))
            except Exception as e:
                logger.warning("Failed to pre-compute signals for %s: %s", pid, e)

        return signal_cache

    def run(self) -> BacktestResult:
        """Run the backtest simulation. Returns BacktestResult."""
        if not self._data:
            return BacktestResult(self.initial_capital, [self.initial_capital], [], [])

        logger.info("Pre-computing model predictions for %d coins...", len(self._data))
        signal_cache = self._precompute_signals()

        cash = self.initial_capital
        positions: dict[str, BacktestPosition] = {}
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = []
        timestamps: list[object] = []

        ref_pid = max(self._data, key=lambda k: len(self._data[k]))
        ref_df = self._data[ref_pid]

        peak_equity = self.initial_capital
        total_steps = len(ref_df) - 100

        for step, i in enumerate(range(100, len(ref_df))):
            current_time = ref_df.index[i]
            timestamps.append(current_time)

            if step % 500 == 0:
                logger.info("Backtest step %d/%d (%.0f%%)", step, total_steps, step / total_steps * 100)

            prices: dict[str, float] = {}
            for pid, df in self._data.items():
                if current_time in df.index:
                    prices[pid] = df.loc[current_time, "close"]
                elif i < len(df):
                    prices[pid] = df.iloc[min(i, len(df) - 1)]["close"]

            # Check stops on existing positions
            to_close = []
            for pid, pos in positions.items():
                price = prices.get(pid, 0)
                if price <= 0:
                    continue

                if price > pos.highest_price:
                    pos.highest_price = price
                    atr_dist = pos.entry_price - pos.stop_loss
                    new_stop = price - atr_dist
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop

                if price <= pos.stop_loss or price >= pos.take_profit:
                    pnl = (price - pos.entry_price) * pos.size
                    is_stop = price <= pos.stop_loss
                    sell_fee = price * pos.size * (self.taker_fee if is_stop else self.maker_fee)
                    pnl -= sell_fee
                    cash += price * pos.size - sell_fee
                    trades.append(BacktestTrade(
                        product_id=pid,
                        side="SELL",
                        entry_price=pos.entry_price,
                        exit_price=price,
                        size=pos.size,
                        pnl=pnl,
                        entry_time=ref_df.index[pos.entry_idx] if pos.entry_idx < len(ref_df.index) else None,
                        exit_time=current_time,
                    ))
                    to_close.append(pid)

            for pid in to_close:
                del positions[pid]

            holdings_value = sum(
                prices.get(pid, 0) * pos.size
                for pid, pos in positions.items()
            )
            equity = cash + holdings_value
            equity_curve.append(equity)

            if equity > peak_equity:
                peak_equity = equity
            drawdown = 1 - equity / peak_equity if peak_equity > 0 else 0

            if equity <= self.initial_capital * (1 - self.risk.max_loss_pct):
                logger.info("Backtest: capital floor hit at step %d", i)
                break

            if drawdown >= 0.30:
                continue
            elif drawdown >= 0.20:
                scale = 0.25
            elif drawdown >= 0.10:
                scale = 0.5
            else:
                scale = 1.0

            if not signal_cache:
                continue

            for pid, df in self._data.items():
                if pid in positions:
                    continue
                if len(positions) >= self.risk.max_open_positions:
                    break
                if i >= len(df):
                    continue
                if pid not in signal_cache:
                    continue

                try:
                    sig_df = signal_cache[pid]
                    if current_time not in sig_df.index and i < len(sig_df):
                        row = sig_df.iloc[i]
                    elif current_time in sig_df.index:
                        row = sig_df.loc[current_time]
                    else:
                        continue

                    xgb_p = float(row["xgb_prob"]) if not np.isnan(row["xgb_prob"]) else 0.5
                    lstm_p = float(row["lstm_pred"]) if not np.isnan(row["lstm_pred"]) else 0.0

                    signal = self.ensemble.combine(pid, xgb_p, lstm_p)

                    if signal.signal != Signal.BUY:
                        continue

                    price = prices.get(pid, 0)
                    if price <= 0:
                        continue

                    atr = df["atr_14"].iloc[min(i, len(df) - 1)] if "atr_14" in df.columns else price * 0.02

                    edge = max(signal.ensemble_score - 0.5, 0.0) * 2
                    kelly = edge * 0.5
                    size_usd = min(
                        equity * kelly * scale,
                        equity * self.risk.max_position_pct * scale,
                    )

                    existing = sum(
                        prices.get(p, 0) * pos.size
                        for p, pos in positions.items()
                        if p == pid
                    )
                    max_conc = equity * self.risk.max_concentration_pct
                    size_usd = min(size_usd, max_conc - existing)

                    if size_usd <= 0 or size_usd > cash:
                        size_usd = min(size_usd, cash * 0.95)
                    if size_usd <= 0:
                        continue

                    round_trip_fee = self.maker_fee + self.taker_fee
                    effective_usd = size_usd / (1 + round_trip_fee)
                    buy_fee = effective_usd * self.maker_fee
                    size_base = effective_usd / price
                    cash -= effective_usd + buy_fee

                    stop = price - atr * self.risk.stop_loss_atr_mult
                    tp = price + atr * self.risk.take_profit_atr_mult

                    positions[pid] = BacktestPosition(
                        product_id=pid,
                        entry_price=price,
                        size=size_base,
                        stop_loss=stop,
                        take_profit=tp,
                        highest_price=price,
                        entry_idx=i,
                    )

                except Exception as e:
                    logger.debug("Backtest signal error for %s: %s", pid, e)

        # Close remaining positions at last known prices
        for pid, pos in positions.items():
            price = 0
            if pid in self._data:
                price = self._data[pid].iloc[-1]["close"]
            if price > 0:
                pnl = (price - pos.entry_price) * pos.size
                cash += price * pos.size
                trades.append(BacktestTrade(
                    product_id=pid,
                    side="SELL",
                    entry_price=pos.entry_price,
                    exit_price=price,
                    size=pos.size,
                    pnl=pnl,
                    entry_time=None,
                    exit_time=None,
                ))

        if not equity_curve:
            equity_curve.append(cash)

        return BacktestResult(
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            trades=trades,
            timestamps=timestamps,
        )
