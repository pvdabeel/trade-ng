"""XGBoost classifier for predicting price direction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss

logger = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {"open", "high", "low", "close", "volume", "target"}


class XGBoostSignalModel:
    """Binary classifier: predicts P(price goes up by > threshold in N candles)."""

    def __init__(
        self,
        target_return_threshold: float = 0.01,
        target_horizon: int = 6,
        model_dir: str = "models",
    ):
        self.threshold = target_return_threshold
        self.horizon = target_horizon
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Create binary target: 1 if future return > threshold, else 0."""
        future_return = df["close"].shift(-self.horizon) / df["close"] - 1
        return (future_return > self.threshold).astype(int)

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in df.columns
            if c not in FEATURE_COLS_EXCLUDE and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]

    def train(
        self,
        df: pd.DataFrame,
        params: dict | None = None,
    ) -> dict:
        """Train on the provided DataFrame using walk-forward validation.

        Returns a dict of metrics from the most recent validation fold.
        """
        target = self.prepare_target(df)
        df = df.copy()
        df["target"] = target

        df.dropna(inplace=True)
        if len(df) < 100:
            logger.warning("Not enough data to train XGBoost (%d rows)", len(df))
            return {}

        self.feature_names = self.get_feature_columns(df)
        X = df[self.feature_names].values
        y = df["target"].values

        # Walk-forward: train on first 80%, validate on last 20%
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "nthread": 1,
        }
        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation metrics
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        val_pred = (val_pred_proba > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_val, val_pred),
            "f1": f1_score(y_val, val_pred, zero_division=0),
            "logloss": log_loss(y_val, val_pred_proba),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "positive_rate": float(y_val.mean()),
        }
        logger.info("XGBoost trained — accuracy=%.3f  f1=%.3f", metrics["accuracy"], metrics["f1"])
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities of positive class."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        cols = [c for c in self.feature_names if c in df.columns]
        if len(cols) < len(self.feature_names):
            missing = set(self.feature_names) - set(cols)
            logger.warning("Missing features for prediction: %s", missing)
        X = df[self.feature_names].fillna(0).values
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict probability for a single data point."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        row = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        return float(self.model.predict_proba(row)[0, 1])

    def feature_importance(self, top_n: int = 20) -> dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        pairs = sorted(zip(self.feature_names, importance), key=lambda x: -x[1])
        return {name: float(imp) for name, imp in pairs[:top_n]}

    def save(self, name: str = "xgboost") -> Path:
        path = self.model_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "horizon": self.horizon,
                },
                f,
            )
        logger.info("XGBoost model saved to %s", path)
        return path

    def load(self, name: str = "xgboost") -> bool:
        path = self.model_dir / f"{name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.threshold = data["threshold"]
        self.horizon = data["horizon"]
        logger.info("XGBoost model loaded from %s", path)
        return True
