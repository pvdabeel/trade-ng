"""Ensemble combiner: merges XGBoost and LSTM outputs into final signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    product_id: str
    signal: Signal
    strength: float  # 0.0 – 1.0
    xgb_prob: float
    lstm_pred: float
    ensemble_score: float


class EnsembleCombiner:
    """Weighted combination of XGBoost probability and LSTM predicted return."""

    def __init__(
        self,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def combine(
        self,
        product_id: str,
        xgb_prob: float,
        lstm_pred: float,
    ) -> TradeSignal:
        """Combine model outputs into a single trade signal.

        Args:
            product_id: e.g. 'BTC-USD'
            xgb_prob: XGBoost P(price up), range [0, 1]
            lstm_pred: LSTM predicted log return (any float)
        """
        lstm_score = _sigmoid(lstm_pred * 100)  # scale and squash to [0,1]

        ensemble_score = (
            self.xgb_weight * xgb_prob + self.lstm_weight * lstm_score
        )

        if ensemble_score >= self.buy_threshold:
            signal = Signal.BUY
        elif ensemble_score <= self.sell_threshold:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        strength = abs(ensemble_score - 0.5) * 2  # 0 at midpoint, 1 at extremes

        return TradeSignal(
            product_id=product_id,
            signal=signal,
            strength=strength,
            xgb_prob=xgb_prob,
            lstm_pred=lstm_pred,
            ensemble_score=ensemble_score,
        )

    def combine_batch(
        self,
        product_ids: list[str],
        xgb_probs: np.ndarray,
        lstm_preds: np.ndarray,
    ) -> list[TradeSignal]:
        signals = []
        for pid, xp, lp in zip(product_ids, xgb_probs, lstm_preds):
            if np.isnan(xp) or np.isnan(lp):
                continue
            signals.append(self.combine(pid, float(xp), float(lp)))
        return signals


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)
