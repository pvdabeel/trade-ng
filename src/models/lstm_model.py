"""LSTM price forecaster using PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

torch.set_num_threads(1)


def _select_device() -> torch.device:
    """Pick the best available accelerator: MPS (Apple Silicon) > CUDA > CPU."""
    import os
    if os.environ.get("PYTORCH_MPS_DISABLE"):
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class LSTMForecaster:
    """LSTM model that predicts next-candle log return."""

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        model_dir: str = "models",
    ):
        self.seq_len = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: LSTMNet | None = None
        self.feature_names: list[str] = []
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None
        self.device = _select_device()

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        exclude = {"open", "high", "low", "close", "volume", "target"}
        return [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(len(X) - self.seq_len):
            Xs.append(X[i : i + self.seq_len])
            ys.append(y[i + self.seq_len])
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame) -> dict:
        """Train the LSTM on feature DataFrame. Returns validation metrics."""
        df = df.copy()
        target = np.log(df["close"] / df["close"].shift(1))
        df["target"] = target
        df.dropna(inplace=True)

        if len(df) < self.seq_len + 50:
            logger.warning("Not enough data to train LSTM (%d rows)", len(df))
            return {}

        self.feature_names = self._get_feature_cols(df)
        X_raw = df[self.feature_names].values
        y_raw = df["target"].values

        # Normalize features
        self.feature_means = np.nanmean(X_raw, axis=0)
        self.feature_stds = np.nanstd(X_raw, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0
        X_norm = (X_raw - self.feature_means) / self.feature_stds
        X_norm = np.nan_to_num(X_norm, 0.0)

        X_seq, y_seq = self._create_sequences(X_norm, y_raw)

        # Walk-forward split: train 80%, val 20%
        split = int(len(X_seq) * 0.8)
        train_ds = SequenceDataset(X_seq[:split], y_seq[:split])
        val_ds = SequenceDataset(X_seq[split:], y_seq[split:])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        input_size = X_seq.shape[2]
        self.model = LSTMNet(input_size, self.hidden_size, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_ds)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    pred = self.model(Xb)
                    val_loss += criterion(pred, yb).item() * len(Xb)
            val_loss /= len(val_ds)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info("LSTM early stopping at epoch %d", epoch + 1)
                    break

        # Directional accuracy on validation set
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(self.device)
                pred = self.model(Xb).cpu().numpy()
                all_preds.extend(pred)
                all_true.extend(yb.numpy())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        directional_acc = np.mean(np.sign(all_preds) == np.sign(all_true))

        metrics = {
            "val_mse": best_val_loss,
            "directional_accuracy": float(directional_acc),
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        }
        logger.info(
            "LSTM trained — val_mse=%.6f  dir_acc=%.3f",
            best_val_loss,
            directional_acc,
        )
        return metrics

    def predict(self, df: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """Predict returns for each row that has sufficient history (batched)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_raw = df[self.feature_names].fillna(0).values
        X_norm = (X_raw - self.feature_means) / self.feature_stds
        X_norm = np.nan_to_num(X_norm, 0.0)

        n = len(X_norm)
        predictions = np.full(n, np.nan)
        if n <= self.seq_len:
            return predictions

        # Build all sequences at once via stride tricks
        seqs = np.stack([X_norm[i - self.seq_len : i] for i in range(self.seq_len, n)])

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                chunk = seqs[start : start + batch_size]
                tensor = torch.tensor(chunk, dtype=torch.float32).to(self.device)
                preds = self.model(tensor).cpu().numpy()
                predictions[self.seq_len + start : self.seq_len + start + len(preds)] = preds

        return predictions

    def predict_single(self, sequence: np.ndarray) -> float:
        """Predict from a single normalized sequence of shape (seq_len, n_features)."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            return float(self.model(tensor).cpu().item())

    def save(self, name: str = "lstm") -> Path:
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = self.model_dir / f"{name}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "feature_names": self.feature_names,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "input_size": len(self.feature_names),
            },
            path,
        )
        logger.info("LSTM model saved to %s", path)
        return path

    def load(self, name: str = "lstm") -> bool:
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            return False
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self.feature_means = data["feature_means"]
        self.feature_stds = data["feature_stds"]
        self.seq_len = data["seq_len"]
        self.hidden_size = data["hidden_size"]
        self.num_layers = data["num_layers"]
        input_size = data["input_size"]
        self.model = LSTMNet(input_size, self.hidden_size, self.num_layers).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        logger.info("LSTM model loaded from %s", path)
        return True
