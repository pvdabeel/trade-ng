"""Training pipeline with walk-forward CV and Optuna hyperparameter tuning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from src.features.cross_asset import add_cross_asset_features
from src.features.market import add_market_features
from src.features.technical import add_technical_features
from src.models.xgboost_model import XGBoostSignalModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_feature_dataframe(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    product_id: str = "",
    volatility_windows: list[int] | None = None,
    correlation_window: int = 30,
) -> pd.DataFrame:
    """Apply all feature engineering steps to a raw OHLCV DataFrame."""
    df = add_technical_features(df.copy())
    df = add_market_features(df, volatility_windows=volatility_windows)
    if btc_df is not None:
        df = add_cross_asset_features(df, btc_df, product_id, correlation_window)
    return df


def walk_forward_cv(
    df: pd.DataFrame,
    model: XGBoostSignalModel,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    params: dict | None = None,
) -> list[dict[str, Any]]:
    """Walk-forward cross-validation to evaluate model without look-ahead bias."""
    target = model.prepare_target(df)
    df = df.copy()
    df["target"] = target
    df.dropna(inplace=True)

    feature_cols = model.get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values
    n = len(X)

    fold_size = n // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_end <= val_start:
            break

        train_start = max(0, train_end - int(fold_size * (1 + train_ratio)))
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        import xgboost as xgb

        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "nthread": 1,
        }
        if params:
            default_params.update(params)

        clf = xgb.XGBClassifier(**default_params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, pred_proba)
        results.append({"fold": i, "logloss": ll, "val_size": len(y_val)})

    return results


def optuna_tune(
    df: pd.DataFrame,
    model: XGBoostSignalModel,
    n_trials: int = 30,
) -> dict:
    """Use Optuna to find optimal XGBoost hyperparameters."""
    target = model.prepare_target(df)
    df = df.copy()
    df["target"] = target
    df.dropna(inplace=True)

    feature_cols = model.get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "nthread": 1,
        }

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_proba = clf.predict_proba(X_val)[:, 1]
        return log_loss(y_val, pred_proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna best logloss: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)
    return study.best_params
