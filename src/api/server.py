"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import create_router
from src.data.coinbase_client import CoinbaseClient
from src.data.database import Database
from src.trading.fx_manager import FXManager
from src.trading.portfolio import PortfolioTracker


def create_app(cfg: dict | None = None) -> FastAPI:
    if cfg is None:
        import yaml
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)

    app = FastAPI(title="Trade-ng Dashboard", version="1.0.0")

    db_path = cfg.get("database", {}).get("path", "data/trade.db")
    client = CoinbaseClient()
    db = Database(db_path)
    portfolio = PortfolioTracker(client, db)

    fx_cfg = cfg.get("fx", {})
    fx_mgr = None
    if fx_cfg.get("home_currency", "").upper() == "EUR":
        trading_cfg = cfg.get("trading", {})
        fx_mgr = FXManager(
            client,
            eurc_product_id=fx_cfg.get("eurc_product_id", "EURC-USDC"),
            max_usd_exposure_pct=fx_cfg.get("max_usd_exposure_pct", 0.30),
            rebalance_threshold_pct=fx_cfg.get("rebalance_threshold_pct", 0.05),
            min_rebalance_usd=fx_cfg.get("min_rebalance_usd", 5.0),
            maker_fee_pct=trading_cfg.get("maker_fee_pct", 0.006),
        )

    trading_cfg_2 = cfg.get("trading", {})
    sync_cfg = cfg.get("sync", {})
    router = create_router(
        db, client, portfolio,
        fx_manager=fx_mgr,
        momentum_scanner=None,
        taker_fee_pct=trading_cfg_2.get("taker_fee_pct", 0.012),
        sync_interval_sec=sync_cfg.get("position_sync_interval_sec", 300),
    )
    app.include_router(router, prefix="/api")

    dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    return app
