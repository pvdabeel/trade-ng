"""CLI entry point for the Trade-ng trading agent."""

from __future__ import annotations

# Must be set before importing torch/xgboost to prevent OpenMP segfaults on macOS
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import asyncio
import datetime as dt
import logging
import signal
import sys
import time

import click
import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.data.coinbase_client import CoinbaseClient
from src.data.database import Database
from src.data.fetcher import CoinUniverse, HistoricalFetcher
from src.data.stream import PriceStream
from src.models.ensemble import EnsembleCombiner, Signal
from src.models.lstm_model import LSTMForecaster
from src.models.trainer import build_feature_dataframe, optuna_tune
from src.models.xgboost_model import XGBoostSignalModel
from src.trading.executor import OrderExecutor
from src.trading.fx_manager import FXManager
from src.trading.momentum import MomentumConfig, MomentumScanner
from src.trading.portfolio import PortfolioTracker
from src.trading.risk import CooldownTracker, RiskConfig, RiskManager, TradingState
from src.trading.signals import SignalGenerator

console = Console()


def _fmt_price(p: float) -> str:
    """Format a price with enough decimals to be meaningful."""
    if p >= 1.0:
        return f"${p:,.2f}"
    if p >= 0.01:
        return f"${p:.4f}"
    if p >= 0.0001:
        return f"${p:.6f}"
    return f"${p:.10f}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trade-ng.log"),
    ],
)
logger = logging.getLogger("trade-ng")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_components(cfg: dict):
    """Construct all system components from config."""
    client = CoinbaseClient()
    db = Database(cfg.get("database", {}).get("path", "data/trade.db"))

    _risk_fields = {f.name for f in RiskConfig.__dataclass_fields__.values()}
    risk_cfg = RiskConfig(**{k: v for k, v in cfg.get("risk", {}).items() if k in _risk_fields})

    xgb_cfg = cfg.get("models", {}).get("xgboost", {})
    xgb_model = XGBoostSignalModel(
        target_return_threshold=xgb_cfg.get("target_return_threshold", 0.01),
        target_horizon=xgb_cfg.get("target_horizon_candles", 6),
    )

    lstm_cfg = cfg.get("models", {}).get("lstm", {})
    lstm_model = LSTMForecaster(
        sequence_length=lstm_cfg.get("sequence_length", 60),
        hidden_size=lstm_cfg.get("hidden_size", 128),
        num_layers=lstm_cfg.get("num_layers", 2),
        learning_rate=lstm_cfg.get("learning_rate", 0.001),
        epochs=lstm_cfg.get("epochs", 50),
        batch_size=lstm_cfg.get("batch_size", 64),
    )

    ens_cfg = cfg.get("models", {}).get("ensemble", {})
    ensemble = EnsembleCombiner(
        buy_threshold=ens_cfg.get("buy_threshold", 0.6),
        sell_threshold=ens_cfg.get("sell_threshold", 0.4),
    )

    trading_cfg = cfg.get("trading", {})
    fx_cfg = cfg.get("fx", {})

    fx_mgr = None
    if fx_cfg.get("home_currency", "").upper() == "EUR":
        fx_mgr = FXManager(
            client,
            eurc_product_id=fx_cfg.get("eurc_product_id", "EURC-USDC"),
            max_usd_exposure_pct=fx_cfg.get("max_usd_exposure_pct", 0.30),
            rebalance_threshold_pct=fx_cfg.get("rebalance_threshold_pct", 0.05),
            min_rebalance_usd=fx_cfg.get("min_rebalance_usd", 5.0),
            maker_fee_pct=trading_cfg.get("maker_fee_pct", 0.006),
            order_timeout_sec=trading_cfg.get("order_timeout_sec", 60),
        )

    risk_mgr = RiskManager(risk_cfg, client, db, fx_manager=fx_mgr)
    risk_cfg_dict = cfg.get("risk", {})
    cooldown_minutes = risk_cfg_dict.get("stop_loss_cooldown_minutes", 30)
    cooldown_tracker = CooldownTracker(default_minutes=cooldown_minutes)
    cooldown_tracker.seed_from_db(db)
    executor_obj = OrderExecutor(
        client, db, risk_mgr,
        order_timeout_sec=trading_cfg.get("order_timeout_sec", 60),
        maker_fee_pct=trading_cfg.get("maker_fee_pct", 0.006),
        taker_fee_pct=trading_cfg.get("taker_fee_pct", 0.012),
        fx_manager=fx_mgr,
        cooldown=cooldown_tracker,
    )

    risk_mgr._on_emergency = executor_obj.emergency_liquidate

    signal_gen = SignalGenerator(xgb_model, lstm_model, ensemble)
    portfolio = PortfolioTracker(client, db)
    universe = CoinUniverse(client)
    fetcher = HistoricalFetcher(client, db)

    return {
        "client": client,
        "db": db,
        "risk_cfg": risk_cfg,
        "risk": risk_mgr,
        "xgb": xgb_model,
        "lstm": lstm_model,
        "ensemble": ensemble,
        "executor": executor_obj,
        "signals": signal_gen,
        "portfolio": portfolio,
        "universe": universe,
        "fetcher": fetcher,
        "fx": fx_mgr,
        "cfg": cfg,
        "cooldown": cooldown_tracker,
    }


@click.group()
def cli():
    """Trade-ng: ML-powered crypto trading agent."""
    pass


@cli.command()
@click.option("--acknowledge-loss", is_flag=True, help="Acknowledge previous losses and restart")
def run(acknowledge_loss: bool):
    """Start live trading."""
    cfg = load_config()
    comp = build_components(cfg)
    client: CoinbaseClient = comp["client"]
    db: Database = comp["db"]
    risk: RiskManager = comp["risk"]
    executor: OrderExecutor = comp["executor"]
    signal_gen: SignalGenerator = comp["signals"]
    xgb: XGBoostSignalModel = comp["xgb"]
    lstm: LSTMForecaster = comp["lstm"]
    fetcher: HistoricalFetcher = comp["fetcher"]
    universe: CoinUniverse = comp["universe"]

    fx_mgr: FXManager | None = comp["fx"]

    console.print("[bold green]Trade-ng Starting...[/bold green]")

    # Check for previous shutdown
    if acknowledge_loss:
        current_val = client.get_portfolio_value()
        db.reset_capital_record()
        db.reset_peak_value(current_val)
        console.print(f"[yellow]Capital floor reset to current value (${current_val:,.2f}).[/yellow]")
    else:
        cap_record = db.get_capital_record()
        if cap_record:
            current_val = client.get_portfolio_value()
            if current_val <= cap_record.capital_floor_usd:
                console.print(
                    "[bold red]Previous emergency shutdown detected.[/bold red]\n"
                    "Portfolio is below capital floor. Use --acknowledge-loss to restart.",
                )
                sys.exit(1)

    # Load or train models
    if not xgb.load():
        console.print("[yellow]No XGBoost model found. Run 'train' first.[/yellow]")
        sys.exit(1)
    if not lstm.load():
        console.print("[yellow]No LSTM model found. Run 'train' first.[/yellow]")
        sys.exit(1)

    # Discover coins
    quote_cfg = cfg.get("trading", {}).get("quote_currency", "auto")
    coin_ids = universe.discover(quote_currency=quote_cfg)
    quote = universe.quote_currency
    console.print(f"Trading {len(coin_ids)} coins (quote: {quote})")

    # Initialize capital tracking
    prices = {pid: client.get_ticker(pid) for pid in coin_ids[:5]}
    initial_value = risk.initialize_capital(prices)
    console.print(f"Portfolio value: ${initial_value:,.2f}")
    console.print(
        f"Capital floor: ${risk._capital_floor:,.2f} "
        f"(max loss {risk.config.max_loss_pct*100:.0f}%)"
    )

    if fx_mgr:
        fx_status = fx_mgr.get_status(initial_value)
        console.print(
            f"EUR/USD: {fx_status.eur_usd_rate:.4f}  "
            f"Portfolio: €{fx_status.portfolio_value_eur:,.2f}  "
            f"USD exposure: {fx_status.usd_exposure_pct*100:.0f}%  "
            f"EURC: {fx_status.eurc_balance:.0f}"
        )

    # Fix any open positions missing stop-loss/take-profit
    for pos in db.get_open_positions():
        if pos.stop_loss and pos.stop_loss > 0:
            continue
        if pos.entry_price and pos.entry_price > 0:
            try:
                cur_price = client.get_ticker(pos.product_id)
            except Exception:
                cur_price = pos.entry_price
            sl = cur_price * 0.92
            tp = max(pos.entry_price, cur_price) * 1.06
            db.update_position(pos.product_id, stop_loss=sl, take_profit=tp, highest_price=cur_price)
            console.print(f"[yellow]Set stops on {pos.product_id}: SL=${sl:.6f} TP=${tp:.6f}[/yellow]")

    # Build momentum scanner (starts only when flag is set)
    mom_cfg_dict = cfg.get("momentum", {})
    mom_cfg = MomentumConfig(**{k: v for k, v in mom_cfg_dict.items() if k in MomentumConfig.__dataclass_fields__})
    trading_cfg_2 = cfg.get("trading", {})
    cooldown_tracker: CooldownTracker = comp["cooldown"]
    momentum = MomentumScanner(
        client=client,
        db=db,
        risk=risk,
        coin_ids=coin_ids,
        config=mom_cfg,
        maker_fee_pct=trading_cfg_2.get("maker_fee_pct", 0.006),
        taker_fee_pct=trading_cfg_2.get("taker_fee_pct", 0.012),
        order_timeout_sec=trading_cfg_2.get("order_timeout_sec", 60),
        fx_manager=fx_mgr,
        cooldown=cooldown_tracker,
    )

    # Start watchdog
    risk.start_watchdog()

    # Set up graceful shutdown
    shutdown = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown
        if shutdown:
            console.print("\n[bold red]Forced exit.[/bold red]")
            risk.stop_watchdog()
            sys.exit(1)
        shutdown = True
        console.print("\n[yellow]Shutting down gracefully (Ctrl+C again to force)...[/yellow]")

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    def _check_momentum_flag() -> None:
        """Start/stop the momentum scanner based on the flag file."""
        if MomentumScanner.is_flag_active() and not momentum.is_running:
            momentum.start()
            console.print("[bold cyan]Momentum scanner activated[/bold cyan]")
        elif not MomentumScanner.is_flag_active() and momentum.is_running:
            momentum.stop()
            console.print("[yellow]Momentum scanner deactivated[/yellow]")

    def _interruptible_sleep(seconds: float) -> None:
        """Sleep in 1-second chunks so Ctrl+C is responsive.
        Also checks the momentum flag every 5 seconds."""
        end = time.time() + seconds
        last_mom_check = 0.0
        while time.time() < end and not shutdown:
            now = time.time()
            if now - last_mom_check >= 5.0:
                _check_momentum_flag()
                last_mom_check = now
            time.sleep(min(1.0, end - time.time()))

    # --- Main trading loop ---
    interval = cfg.get("trading", {}).get("trading_interval_sec", 300)
    granularity = cfg.get("trading", {}).get("candle_granularity", "ONE_HOUR")

    console.print(f"[green]Trading loop started (interval={interval}s)[/green]")

    while not shutdown:
        try:
            loop_start = time.time()

            # Check momentum flag at loop start
            _check_momentum_flag()

            # Get current prices (only for open positions + a sample)
            current_prices = {}
            open_pids = {p.product_id for p in db.get_open_positions()}
            price_pids = list(open_pids) + coin_ids[:10]
            for pid in price_pids:
                if shutdown:
                    break
                try:
                    current_prices[pid] = client.get_ticker(pid)
                except Exception:
                    pass

            if shutdown:
                break

            # Check stops on existing positions
            closed = executor.check_stops(current_prices)
            if closed:
                console.print(f"[yellow]Closed positions: {closed}[/yellow]")

            # Check risk state
            portfolio_value = client.get_portfolio_value(current_prices)
            if risk.state in (TradingState.EMERGENCY, TradingState.SHUTDOWN):
                console.print("[bold red]EMERGENCY STATE — shutting down[/bold red]")
                executor.emergency_liquidate()
                break

            if risk.state in (TradingState.HALTED_DAILY, TradingState.HALTED_HOURLY):
                console.print(f"[yellow]Trading halted: {risk.state.value}[/yellow]")
                _interruptible_sleep(interval)
                continue

            btc_candle_id = "BTC-USD"
            btc_df = fetcher.load_candles(btc_candle_id, granularity, lookback_days=90)

            for pid in coin_ids:
                if shutdown:
                    break

                try:
                    candle_id = pid.replace("-USDC", "-USD") if quote == "USDC" else pid
                    df = fetcher.load_candles(candle_id, granularity, lookback_days=90)
                    if len(df) < 100:
                        continue

                    feat_cfg = cfg.get("features", {})
                    df = build_feature_dataframe(
                        df, btc_df, candle_id,
                        volatility_windows=feat_cfg.get("volatility_windows", [5, 10, 20, 50]),
                        correlation_window=feat_cfg.get("correlation_window", 30),
                    )

                    signal_result = signal_gen.generate(pid, df)
                    if signal_result is None or signal_result.signal == Signal.HOLD:
                        continue

                    if pid not in current_prices:
                        try:
                            current_prices[pid] = client.get_ticker(pid)
                        except Exception:
                            continue

                    price = current_prices.get(pid, 0)
                    atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else price * 0.02

                    executor.execute_signal(
                        signal=signal_result,
                        portfolio_value=portfolio_value,
                        current_price=price,
                        atr=atr,
                        coin_universe=coin_ids,
                    )

                except Exception as e:
                    logger.error("Error processing %s: %s", pid, e)

            # Momentum scanner: auto-start/stop based on flag file
            if not shutdown:
                _check_momentum_flag()

            # FX rebalance
            if fx_mgr and not shutdown:
                try:
                    fx_mgr.rebalance_to_eurc()
                    fx_s = fx_mgr.get_status(portfolio_value)
                    logger.info(
                        "FX: EUR/USD=%.4f  portfolio=€%.2f  USD exposure=%.0f%%  EURC=%.0f",
                        fx_s.eur_usd_rate,
                        fx_s.portfolio_value_eur,
                        fx_s.usd_exposure_pct * 100,
                        fx_s.eurc_balance,
                    )
                except Exception as e:
                    logger.error("FX rebalance error: %s", e)

            # Sleep until next interval (interruptible)
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0 and not shutdown:
                _interruptible_sleep(sleep_time)

        except Exception as e:
            logger.error("Trading loop error: %s", e)
            _interruptible_sleep(10)

    # Cleanup
    if momentum.is_running or momentum.is_monitoring:
        momentum.stop_all()
    risk.stop_watchdog()
    console.print("[bold]Trade-ng stopped.[/bold]")


@cli.command()
@click.option("--tune/--no-tune", default=False, help="Run Optuna hyperparameter tuning")
@click.option("--trials", default=30, help="Number of Optuna trials")
def train(tune: bool, trials: int):
    """Train ML models on historical data."""
    cfg = load_config()
    comp = build_components(cfg)
    client: CoinbaseClient = comp["client"]
    fetcher: HistoricalFetcher = comp["fetcher"]
    universe: CoinUniverse = comp["universe"]
    xgb: XGBoostSignalModel = comp["xgb"]
    lstm: LSTMForecaster = comp["lstm"]

    console.print("[bold]Training ML models...[/bold]")

    coin_ids = universe.discover()
    console.print(f"Discovered {len(coin_ids)} coins")

    granularity = cfg.get("trading", {}).get("candle_granularity", "ONE_HOUR")
    lookback = cfg.get("models", {}).get("xgboost", {}).get("lookback_days", 90)
    feat_cfg = cfg.get("features", {})

    # Load BTC data first (needed for cross-asset features); reuses DB cache
    console.print("Loading BTC-USD historical data...")
    btc_df = fetcher.load_candles("BTC-USD", granularity, lookback)

    # Collect training data from all coins (loads from DB, fetches only if missing)
    all_dfs = []
    with console.status("Loading and processing coin data..."):
        for pid in coin_ids:
            try:
                df = fetcher.load_candles(pid, granularity, lookback)
                if len(df) < 200:
                    continue
                df = build_feature_dataframe(
                    df, btc_df, pid,
                    volatility_windows=feat_cfg.get("volatility_windows", [5, 10, 20, 50]),
                    correlation_window=feat_cfg.get("correlation_window", 30),
                )
                df["_product_id"] = pid
                all_dfs.append(df)
                console.print(f"  {pid}: {len(df)} candles")
            except Exception as e:
                console.print(f"  [red]{pid}: {e}[/red]")

    if not all_dfs:
        console.print("[red]No data available for training.[/red]")
        return

    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=False)
    combined.drop(columns=["_product_id"], inplace=True, errors="ignore")

    # Optuna tuning
    best_params = None
    if tune:
        console.print(f"\n[bold]Running Optuna ({trials} trials)...[/bold]")
        best_params = optuna_tune(combined, xgb, n_trials=trials)
        console.print(f"Best params: {best_params}")

    # Train XGBoost
    console.print("\n[bold]Training XGBoost...[/bold]")
    xgb_metrics = xgb.train(combined, params=best_params)
    if xgb_metrics:
        xgb.save()
        console.print(f"  Accuracy: {xgb_metrics['accuracy']:.3f}")
        console.print(f"  F1 Score: {xgb_metrics['f1']:.3f}")
        top_features = xgb.feature_importance(10)
        console.print("  Top features:")
        for name, imp in top_features.items():
            console.print(f"    {name}: {imp:.4f}")

    # Train LSTM
    console.print("\n[bold]Training LSTM...[/bold]")
    lstm_metrics = lstm.train(combined)
    if lstm_metrics:
        lstm.save()
        console.print(f"  Val MSE: {lstm_metrics['val_mse']:.6f}")
        console.print(f"  Directional Accuracy: {lstm_metrics['directional_accuracy']:.3f}")

    console.print("\n[bold green]Training complete![/bold green]")


@cli.command()
def status():
    """Show current portfolio and positions."""
    cfg = load_config()
    comp = build_components(cfg)
    portfolio: PortfolioTracker = comp["portfolio"]
    risk: RiskManager = comp["risk"]
    db: Database = comp["db"]

    try:
        summary = portfolio.get_summary()
    except Exception as e:
        console.print(f"[red]Error fetching portfolio: {e}[/red]")
        return

    fx_mgr: FXManager | None = comp["fx"]

    console.print(f"\n[bold]Portfolio Summary[/bold]")
    console.print(f"  Total Value:  ${summary.total_value_usd:,.2f}")
    console.print(f"  Cash:         ${summary.cash_usd:,.2f}")
    console.print(f"  Holdings:     ${summary.holdings_value_usd:,.2f}")
    console.print(f"  Positions:    {summary.num_open_positions}")
    console.print(f"  Unrealized:   ${summary.total_unrealized_pnl:,.2f}")

    if fx_mgr:
        fx_s = fx_mgr.get_status(summary.total_value_usd)
        console.print(f"\n[bold]Currency (EUR)[/bold]")
        console.print(f"  EUR/USD Rate:    {fx_s.eur_usd_rate:.4f}")
        console.print(f"  Value in EUR:    €{fx_s.portfolio_value_eur:,.2f}")
        console.print(f"  EURC Balance:    {fx_s.eurc_balance:.0f} EURC")
        console.print(f"  USDC Balance:    ${fx_s.usdc_balance:,.2f}")
        exp_color = "green" if fx_s.usd_exposure_pct <= 0.35 else "yellow" if fx_s.usd_exposure_pct <= 0.50 else "red"
        console.print(f"  USD Exposure:    [{exp_color}]{fx_s.usd_exposure_pct*100:.0f}%[/{exp_color}]")

    cap_record = db.get_capital_record()
    if cap_record:
        console.print(f"\n[bold]Capital Protection[/bold]")
        console.print(f"  Initial Capital: ${cap_record.initial_capital_usd:,.2f}")
        console.print(f"  Capital Floor:   ${cap_record.capital_floor_usd:,.2f}")
        drawdown = (
            (1 - summary.total_value_usd / cap_record.initial_capital_usd) * 100
            if cap_record.initial_capital_usd > 0
            else 0
        )
        color = "green" if drawdown < 10 else "yellow" if drawdown < 30 else "red"
        console.print(f"  Drawdown:        [{color}]{drawdown:.1f}%[/{color}]")

    if summary.positions:
        table = Table(title="Open Positions")
        table.add_column("Coin")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Target", justify="right")

        for pos in summary.positions:
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                pos.product_id,
                _fmt_price(pos.entry_price),
                _fmt_price(pos.current_price),
                f"{pos.size:,.6f}",
                f"[{pnl_color}]${pos.unrealized_pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pos.unrealized_pnl_pct:+.1f}%[/{pnl_color}]",
                _fmt_price(pos.stop_loss),
                _fmt_price(pos.take_profit),
            )
        console.print(table)

    # Recent trades
    trades = db.get_recent_trades(10)
    if trades:
        trade_table = Table(title="Recent Trades")
        trade_table.add_column("Time")
        trade_table.add_column("Coin")
        trade_table.add_column("Side")
        trade_table.add_column("Price", justify="right")
        trade_table.add_column("Size", justify="right")
        trade_table.add_column("Status")

        for t in trades:
            side_color = "green" if t.side == "BUY" else "red"
            trade_table.add_row(
                t.created_at.strftime("%m-%d %H:%M") if t.created_at else "",
                t.product_id,
                f"[{side_color}]{t.side}[/{side_color}]",
                _fmt_price(t.price),
                f"{t.size:,.6f}",
                t.status,
            )
        console.print(trade_table)


@cli.command()
def dashboard():
    """Launch the web monitoring dashboard."""
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8080)

    console.print(f"[bold]Starting dashboard at http://{host}:{port}[/bold]")

    import uvicorn
    from src.api.server import create_app

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--days", default=90, help="Lookback period in days")
def backtest(days: int):
    """Run backtesting on historical data."""
    cfg = load_config()
    comp = build_components(cfg)
    fetcher: HistoricalFetcher = comp["fetcher"]
    universe: CoinUniverse = comp["universe"]
    xgb: XGBoostSignalModel = comp["xgb"]
    lstm: LSTMForecaster = comp["lstm"]

    if not xgb.load():
        console.print("[red]No XGBoost model. Run 'train' first.[/red]")
        return
    if not lstm.load():
        console.print("[red]No LSTM model. Run 'train' first.[/red]")
        return

    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import compute_metrics

    console.print(f"[bold]Running backtest ({days} days)...[/bold]")

    coin_ids = universe.discover()
    granularity = cfg.get("trading", {}).get("candle_granularity", "ONE_HOUR")
    feat_cfg = cfg.get("features", {})

    btc_df = fetcher.load_candles("BTC-USD", granularity, days)

    trading_cfg = cfg.get("trading", {})
    engine = BacktestEngine(
        initial_capital=10000.0,
        risk_config=comp["risk_cfg"],
        xgb_model=xgb,
        lstm_model=lstm,
        ensemble=comp["ensemble"],
        maker_fee_pct=trading_cfg.get("maker_fee_pct", 0.006),
        taker_fee_pct=trading_cfg.get("taker_fee_pct", 0.012),
    )

    for pid in coin_ids[:20]:
        try:
            df = fetcher.load_candles(pid, granularity, days)
            if len(df) < 200:
                continue
            df = build_feature_dataframe(
                df, btc_df, pid,
                volatility_windows=feat_cfg.get("volatility_windows"),
                correlation_window=feat_cfg.get("correlation_window", 30),
            )
            engine.add_data(pid, df)
        except Exception as e:
            logger.warning("Skipping %s: %s", pid, e)

    results = engine.run()
    metrics = compute_metrics(results)

    console.print(f"\n[bold]Backtest Results[/bold]")
    console.print(f"  Initial Capital:  ${metrics['initial_capital']:,.2f}")
    console.print(f"  Final Value:      ${metrics['final_value']:,.2f}")
    console.print(f"  Total Return:     {metrics['total_return_pct']:.1f}%")
    console.print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    console.print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    console.print(f"  Win Rate:         {metrics['win_rate_pct']:.1f}%")
    console.print(f"  Total Trades:     {metrics['total_trades']}")
    console.print(f"  Avg Trade P&L:    ${metrics['avg_trade_pnl']:,.2f}")


@cli.group()
def momentum():
    """On-demand momentum trading strategy."""
    pass


@momentum.command("start")
def momentum_start():
    """Activate the momentum scanner (takes effect in the running bot)."""
    MomentumScanner.set_flag(True)
    console.print("[bold green]Momentum scanner activated.[/bold green]")
    console.print("The running bot will pick this up within 30 seconds.")


@momentum.command("stop")
def momentum_stop():
    """Deactivate the momentum scanner."""
    MomentumScanner.set_flag(False)
    console.print("[yellow]Momentum scanner deactivated.[/yellow]")
    console.print("The running bot will stop scanning within 30 seconds.")


@momentum.command("status")
def momentum_status():
    """Show momentum scanner status and positions."""
    is_active = MomentumScanner.is_flag_active()
    state_str = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]INACTIVE[/dim]"
    console.print(f"\n[bold]Momentum Strategy[/bold]  {state_str}")

    cfg = load_config()
    comp = build_components(cfg)
    db: Database = comp["db"]

    positions = db.get_open_positions(strategy="momentum")
    console.print(f"  Open positions: {len(positions)}")

    if positions:
        table = Table(title="Momentum Positions")
        table.add_column("Coin")
        table.add_column("Entry", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Opened", justify="right")

        for pos in positions:
            age = ""
            if pos.opened_at:
                delta = dt.datetime.utcnow() - pos.opened_at
                age = f"{int(delta.total_seconds() / 60)}min"
            table.add_row(
                pos.product_id,
                _fmt_price(pos.entry_price),
                f"{pos.size:,.6f}",
                _fmt_price(pos.stop_loss) if pos.stop_loss else "--",
                _fmt_price(pos.take_profit) if pos.take_profit else "--",
                age,
            )
        console.print(table)

    # Recent momentum trades
    trades = db.get_recent_trades(20)
    mom_trades = [t for t in trades if getattr(t, "strategy", "ml") == "momentum"]
    if mom_trades:
        trade_table = Table(title="Recent Momentum Trades")
        trade_table.add_column("Time")
        trade_table.add_column("Coin")
        trade_table.add_column("Side")
        trade_table.add_column("Price", justify="right")
        trade_table.add_column("Fee", justify="right")
        trade_table.add_column("Status")
        trade_table.add_column("Notes")

        for t in mom_trades[:10]:
            side_color = "green" if t.side == "BUY" else "red"
            trade_table.add_row(
                t.created_at.strftime("%H:%M:%S") if t.created_at else "",
                t.product_id,
                f"[{side_color}]{t.side}[/{side_color}]",
                _fmt_price(t.price),
                f"${t.fee:.4f}" if t.fee else "--",
                t.status,
                t.notes or "",
            )
        console.print(trade_table)


def main():
    cli()


if __name__ == "__main__":
    main()
