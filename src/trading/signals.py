"""Signal generation: orchestrates models to produce actionable trade signals."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.ensemble import EnsembleCombiner, TradeSignal
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostSignalModel

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Runs both models on current data and produces ensemble trade signals."""

    def __init__(
        self,
        xgb_model: XGBoostSignalModel,
        lstm_model: LSTMForecaster,
        ensemble: EnsembleCombiner,
    ):
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.ensemble = ensemble

    def generate(
        self,
        product_id: str,
        df: pd.DataFrame,
    ) -> TradeSignal | None:
        """Generate a signal for one product given its feature DataFrame.

        Uses the last row of the DataFrame for prediction.
        """
        if len(df) < self.lstm.seq_len + 1:
            return None

        try:
            xgb_probs = self.xgb.predict(df)
            xgb_prob = float(xgb_probs[-1]) if len(xgb_probs) > 0 else 0.5
        except Exception as e:
            logger.warning("XGBoost prediction failed for %s: %s", product_id, e)
            xgb_prob = 0.5

        try:
            lstm_preds = self.lstm.predict(df)
            lstm_pred = float(lstm_preds[-1]) if len(lstm_preds) > 0 and not np.isnan(lstm_preds[-1]) else 0.0
        except Exception as e:
            logger.warning("LSTM prediction failed for %s: %s", product_id, e)
            lstm_pred = 0.0

        signal = self.ensemble.combine(product_id, xgb_prob, lstm_pred)
        logger.debug(
            "%s signal=%s  strength=%.3f  xgb=%.3f  lstm=%.5f  ensemble=%.3f",
            product_id,
            signal.signal.value,
            signal.strength,
            xgb_prob,
            lstm_pred,
            signal.ensemble_score,
        )
        return signal

    def generate_batch(
        self,
        data: dict[str, pd.DataFrame],
    ) -> list[TradeSignal]:
        """Generate signals for multiple products at once."""
        signals = []
        for product_id, df in data.items():
            sig = self.generate(product_id, df)
            if sig is not None:
                signals.append(sig)
        return signals
