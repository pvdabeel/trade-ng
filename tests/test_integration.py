"""End-to-end integration tests.

Exercises the full pipeline — features, models, ensemble, risk, executor,
backtest, database, portfolio, and API — using synthetic data so no live
Coinbase connection is required.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic OHLCV helper
# ---------------------------------------------------------------------------

def _synth_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 10)
    high = close * (1 + np.abs(rng.randn(n) * 0.01))
    low = close * (1 - np.abs(rng.randn(n) * 0.01))
    open_ = low + (high - low) * rng.rand(n)
    volume = np.abs(rng.randn(n) * 1000) + 500

    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. Full feature pipeline
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_build_feature_dataframe(self):
        from src.models.trainer import build_feature_dataframe

        btc = _synth_ohlcv(500, seed=1)
        eth = _synth_ohlcv(500, seed=2)

        df = build_feature_dataframe(eth, btc, "ETH-USD")

        assert len(df) == 500
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
        assert "log_return_1" in df.columns
        assert "btc_correlation" in df.columns
        assert df.select_dtypes(include=[np.number]).shape[1] >= 40


# ---------------------------------------------------------------------------
# 2. XGBoost train + predict round-trip
# ---------------------------------------------------------------------------

class TestXGBoostRoundTrip:
    def test_train_predict_save_load(self, tmp_path):
        from src.models.trainer import build_feature_dataframe
        from src.models.xgboost_model import XGBoostSignalModel

        btc = _synth_ohlcv(500, seed=1)
        eth = _synth_ohlcv(500, seed=2)
        df = build_feature_dataframe(eth, btc, "ETH-USD")

        model = XGBoostSignalModel(model_dir=str(tmp_path))
        metrics = model.train(df)

        assert metrics, "Training should return metrics"
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["train_size"] > 0

        probs = model.predict(df)
        assert len(probs) == len(df)
        assert np.all((probs >= 0) & (probs <= 1))

        model.save("test_xgb")
        model2 = XGBoostSignalModel(model_dir=str(tmp_path))
        assert model2.load("test_xgb")
        probs2 = model2.predict(df)
        np.testing.assert_array_almost_equal(probs, probs2)


# ---------------------------------------------------------------------------
# 3. LSTM train + predict round-trip
# ---------------------------------------------------------------------------

class TestLSTMRoundTrip:
    def test_train_predict_save_load(self, tmp_path):
        """Run in subprocess to avoid XGBoost OpenMP / PyTorch thread conflict."""
        import subprocess, sys, textwrap

        script = textwrap.dedent(f"""\
            import os, sys
            os.environ["PYTORCH_MPS_DISABLE"] = "1"
            import numpy as np, pandas as pd, torch
            sys.path.insert(0, os.getcwd())

            rng = np.random.RandomState
            def ohlcv(n, seed):
                r = rng(seed); c = np.maximum(100+np.cumsum(r.randn(n)*0.5),10)
                h = c*(1+np.abs(r.randn(n)*0.01)); l = c*(1-np.abs(r.randn(n)*0.01))
                return pd.DataFrame({{"open":l+(h-l)*r.rand(n),"high":h,"low":l,"close":c,
                    "volume":np.abs(r.randn(n)*1000)+500}},
                    index=pd.date_range("2025-01-01",periods=n,freq="h"))

            from src.models.trainer import build_feature_dataframe
            from src.models.lstm_model import LSTMForecaster

            df = build_feature_dataframe(ohlcv(150,2), ohlcv(150,1), "ETH-USD")
            m = LSTMForecaster(sequence_length=20,hidden_size=16,num_layers=1,
                               epochs=2,batch_size=64,model_dir="{tmp_path}")
            m.device = torch.device("cpu")
            met = m.train(df)
            assert met and met["val_mse"] >= 0, f"bad metrics {{met}}"
            p = m.predict(df); assert np.sum(~np.isnan(p)) > 0
            m.save("test_lstm")
            m2 = LSTMForecaster(model_dir="{tmp_path}")
            m2.device = torch.device("cpu")
            assert m2.load("test_lstm")
            p2 = m2.predict(df); assert np.sum(~np.isnan(p2)) > 0
            print("OK")
        """)
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=os.getcwd(),
        )
        assert r.returncode == 0, f"LSTM test failed:\n{r.stderr}"
        assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# 4. Ensemble signal generation
# ---------------------------------------------------------------------------

class TestEnsembleSignals:
    def test_combine_produces_valid_signals(self):
        from src.models.ensemble import EnsembleCombiner, Signal

        ens = EnsembleCombiner(buy_threshold=0.6, sell_threshold=0.4)

        buy_sig = ens.combine("BTC-USD", xgb_prob=0.8, lstm_pred=0.005)
        assert buy_sig.signal == Signal.BUY
        assert 0 <= buy_sig.strength <= 1

        sell_sig = ens.combine("ETH-USD", xgb_prob=0.2, lstm_pred=-0.005)
        assert sell_sig.signal == Signal.SELL

        hold_sig = ens.combine("SOL-USD", xgb_prob=0.5, lstm_pred=0.0)
        assert hold_sig.signal == Signal.HOLD

    def test_batch_combine(self):
        from src.models.ensemble import EnsembleCombiner

        ens = EnsembleCombiner()
        signals = ens.combine_batch(
            ["BTC-USD", "ETH-USD"],
            np.array([0.8, 0.3]),
            np.array([0.002, -0.003]),
        )
        assert len(signals) == 2


# ---------------------------------------------------------------------------
# 5. Database round-trip
# ---------------------------------------------------------------------------

class TestDatabaseRoundTrip:
    def test_capital_record_immutable(self, tmp_path):
        from src.data.database import Database

        db = Database(str(tmp_path / "test.db"))

        rec = db.set_capital_record(10000, 0.50)
        assert rec.initial_capital_usd == 10000
        assert rec.capital_floor_usd == 5000

        rec2 = db.set_capital_record(99999, 0.99)
        assert rec2.initial_capital_usd == 10000, "Capital record must be immutable"

    def test_position_lifecycle(self, tmp_path):
        from src.data.database import Database, Position

        db = Database(str(tmp_path / "test.db"))

        pos = Position(
            product_id="BTC-USD", entry_price=50000, size=0.1,
            stop_loss=48000, take_profit=55000, highest_price=50000,
        )
        db.save_position(pos)

        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].product_id == "BTC-USD"

        db.close_position("BTC-USD", pnl=500)
        assert len(db.get_open_positions()) == 0

    def test_trade_save_and_retrieve(self, tmp_path):
        from src.data.database import Database, Trade

        db = Database(str(tmp_path / "test.db"))

        trade = Trade(
            order_id="order-1", product_id="ETH-USD", side="BUY",
            size=1.0, price=3000, status="FILLED", signal_strength=0.75,
        )
        db.save_trade(trade)

        trades = db.get_recent_trades(10)
        assert len(trades) == 1
        assert trades[0].order_id == "order-1"

    def test_snapshot_and_peak(self, tmp_path):
        from src.data.database import Database, PortfolioSnapshot

        db = Database(str(tmp_path / "test.db"))

        for val in [10000, 11000, 10500]:
            snap = PortfolioSnapshot(
                total_value_usd=val, cash_usd=val * 0.5,
                holdings_value_usd=val * 0.5, num_positions=2,
                drawdown_pct=0.0,
            )
            db.save_snapshot(snap)

        assert db.get_peak_value() == 11000


# ---------------------------------------------------------------------------
# 6. Backtest engine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_backtest_runs_without_crash(self, tmp_path):
        """Run in subprocess to avoid XGBoost OpenMP / PyTorch thread conflict."""
        import subprocess, sys, textwrap

        script = textwrap.dedent(f"""\
            import os, sys
            os.environ["PYTORCH_MPS_DISABLE"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            import torch
            torch.set_num_threads(1)
            import numpy as np, pandas as pd
            sys.path.insert(0, os.getcwd())

            rng = np.random.RandomState
            def ohlcv(n, seed):
                r = rng(seed); c = np.maximum(100+np.cumsum(r.randn(n)*0.5),10)
                h = c*(1+np.abs(r.randn(n)*0.01)); l = c*(1-np.abs(r.randn(n)*0.01))
                return pd.DataFrame({{"open":l+(h-l)*r.rand(n),"high":h,"low":l,"close":c,
                    "volume":np.abs(r.randn(n)*1000)+500}},
                    index=pd.date_range("2025-01-01",periods=n,freq="h"))

            from src.models.trainer import build_feature_dataframe
            from src.models.lstm_model import LSTMForecaster
            from src.models.ensemble import EnsembleCombiner
            from src.backtest.engine import BacktestEngine
            from src.backtest.metrics import compute_metrics
            from src.trading.risk import RiskConfig
            from src.models.xgboost_model import XGBoostSignalModel

            btc = ohlcv(200,1); eth = ohlcv(200,2)
            btc_f = build_feature_dataframe(btc, btc, "BTC-USD")
            eth_f = build_feature_dataframe(eth, btc, "ETH-USD")

            lstm = LSTMForecaster(sequence_length=20,hidden_size=16,num_layers=1,
                                  epochs=2,batch_size=64,model_dir="{tmp_path}")
            lstm.device = torch.device("cpu")
            lstm.train(btc_f)

            xgb = XGBoostSignalModel(model_dir="{tmp_path}")
            xgb.train(btc_f)

            engine = BacktestEngine(initial_capital=10000,risk_config=RiskConfig(),
                                    xgb_model=xgb,lstm_model=lstm,ensemble=EnsembleCombiner())
            engine.add_data("BTC-USD", btc_f)
            engine.add_data("ETH-USD", eth_f)
            result = engine.run()
            m = compute_metrics(result)
            assert m["initial_capital"] == 10000
            assert m["final_value"] > 0
            assert len(result.equity_curve) > 0
            assert m["max_drawdown_pct"] >= 0
            print("OK")
        """)
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
            cwd=os.getcwd(),
        )
        assert r.returncode == 0, f"Backtest test failed:\n{r.stderr}"
        assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# 7. API endpoints
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def test_health_endpoint(self, tmp_path):
        from fastapi.testclient import TestClient

        from src.data.coinbase_client import CoinbaseClient
        from src.data.database import Database
        from src.trading.portfolio import PortfolioTracker
        from src.api.routes import create_router
        from fastapi import FastAPI

        db = Database(str(tmp_path / "api_test.db"))
        client = CoinbaseClient.__new__(CoinbaseClient)
        portfolio = PortfolioTracker(client, db)

        app = FastAPI()
        router = create_router(db, client, portfolio)
        app.include_router(router, prefix="/api")

        tc = TestClient(app)
        resp = tc.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_equity_endpoint_empty(self, tmp_path):
        from fastapi.testclient import TestClient

        from src.data.coinbase_client import CoinbaseClient
        from src.data.database import Database
        from src.trading.portfolio import PortfolioTracker
        from src.api.routes import create_router
        from fastapi import FastAPI

        db = Database(str(tmp_path / "api_test.db"))
        client = CoinbaseClient.__new__(CoinbaseClient)
        portfolio = PortfolioTracker(client, db)

        app = FastAPI()
        router = create_router(db, client, portfolio)
        app.include_router(router, prefix="/api")

        tc = TestClient(app)
        resp = tc.get("/api/equity?hours=24")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_trades_endpoint_empty(self, tmp_path):
        from fastapi.testclient import TestClient

        from src.data.coinbase_client import CoinbaseClient
        from src.data.database import Database
        from src.trading.portfolio import PortfolioTracker
        from src.api.routes import create_router
        from fastapi import FastAPI

        db = Database(str(tmp_path / "api_test.db"))
        client = CoinbaseClient.__new__(CoinbaseClient)
        portfolio = PortfolioTracker(client, db)

        app = FastAPI()
        router = create_router(db, client, portfolio)
        app.include_router(router, prefix="/api")

        tc = TestClient(app)
        resp = tc.get("/api/trades?limit=10")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_risk_endpoint(self, tmp_path):
        from fastapi.testclient import TestClient

        from src.data.coinbase_client import CoinbaseClient
        from src.data.database import Database, PortfolioSnapshot
        from src.trading.portfolio import PortfolioTracker
        from src.api.routes import create_router
        from fastapi import FastAPI

        db = Database(str(tmp_path / "api_test.db"))
        db.set_capital_record(10000, 0.50)
        snap = PortfolioSnapshot(
            total_value_usd=9500, cash_usd=5000,
            holdings_value_usd=4500, num_positions=2,
            drawdown_pct=0.05,
        )
        db.save_snapshot(snap)

        client = CoinbaseClient.__new__(CoinbaseClient)
        portfolio = PortfolioTracker(client, db)

        app = FastAPI()
        router = create_router(db, client, portfolio)
        app.include_router(router, prefix="/api")

        tc = TestClient(app)
        resp = tc.get("/api/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initial_capital"] == 10000
        assert data["capital_floor"] == 5000
