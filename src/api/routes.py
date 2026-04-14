"""REST API endpoints for the dashboard."""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter

from src.data.coinbase_client import CoinbaseClient
from src.data.database import Database, Position, Trade
from src.trading.fx_manager import FXManager
from src.trading.momentum import MomentumScanner, UNTRADEABLE_FILE
from src.trading.portfolio import PortfolioTracker

logger = logging.getLogger("trade-ng.api")

_DECISION_LOG = Path("data/decision_log.json")


def _record_decision_yield(
    product_id: str,
    entry_price: float,
    size: float,
    opened_at: dt.datetime | None,
    net_pnl: float,
    exit_fee: float,
    reason: str,
) -> None:
    """Update the decision log file with realized yield for an enter decision."""
    try:
        if not _DECISION_LOG.exists():
            return
        records = json.loads(_DECISION_LOG.read_text(encoding="utf-8"))
        entry_value = entry_price * size
        if entry_value <= 0:
            return
        pnl_pct = round(net_pnl / entry_value, 4)
        hold_sec = 0
        if opened_at:
            hold_sec = int((dt.datetime.utcnow() - opened_at).total_seconds())
        fee_pct = round((exit_fee * 2) / entry_value, 4)

        matched = False
        for rec in reversed(records):
            if (rec.get("product_id") == product_id
                    and rec.get("decision") == "enter"
                    and rec.get("realized_pnl_pct") is None):
                rec["realized_pnl_pct"] = pnl_pct
                rec["exit_reason"] = reason
                rec["hold_seconds"] = hold_sec
                rec["entry_fee_pct"] = fee_pct
                matched = True
                break
        if matched:
            _DECISION_LOG.write_text(json.dumps(records), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not record decision yield for %s: %s", product_id, exc)


def create_router(
    db: Database,
    client: CoinbaseClient,
    portfolio: PortfolioTracker,
    fx_manager: FXManager | None = None,
    momentum_scanner: MomentumScanner | None = None,
    taker_fee_pct: float = 0.012,
    sync_interval_sec: int = 300,
) -> APIRouter:
    router = APIRouter()

    @router.get("/portfolio")
    def get_portfolio():
        summary = portfolio.get_summary()
        cap = db.get_capital_record()
        result = {
            "total_value": summary.total_value_usd,
            "cash": summary.cash_usd,
            "holdings": summary.holdings_value_usd,
            "num_positions": summary.num_open_positions,
            "unrealized_pnl": summary.total_unrealized_pnl,
            "initial_capital": cap.initial_capital_usd if cap else 0,
            "capital_floor": cap.capital_floor_usd if cap else 0,
            "positions": [p.to_dict() for p in summary.positions],
        }
        if fx_manager:
            fx_s = fx_manager.get_status(summary.total_value_usd)
            result["fx"] = {
                "eur_usd_rate": fx_s.eur_usd_rate,
                "total_value_eur": fx_s.portfolio_value_eur,
                "eurc_balance": fx_s.eurc_balance,
                "usdc_balance": fx_s.usdc_balance,
                "usd_exposure_pct": fx_s.usd_exposure_pct,
                "maker_fee_pct": fx_manager.maker_fee,
            }
        return result

    @router.get("/trades")
    def get_trades(limit: int = 50):
        trades = db.get_recent_trades(limit)

        # Seed entry prices from the positions table so sells whose
        # matching buy is outside the recent-trades window still get P&L.
        last_buy: dict[str, float] = {}
        for pos in db.get_closed_positions():
            last_buy[pos.product_id] = pos.entry_price
        for pos in db.get_open_positions():
            last_buy[pos.product_id] = pos.entry_price

        # Walk trades chronologically (oldest → newest).  Each BUY sets
        # the entry price; each SELL uses the most recent preceding BUY.
        sell_pnl: dict[str, float] = {}
        for t in reversed(trades):
            if t.side == "BUY" and t.status in ("FILLED", "PENDING"):
                last_buy[t.product_id] = t.price
            elif t.side == "SELL" and t.status == "FILLED":
                entry = last_buy.get(t.product_id)
                if entry is not None:
                    sell_fee = t.fee or 0
                    sell_pnl[t.order_id] = (t.price - entry) * t.size - sell_fee

        result = []
        for t in trades:
            value = t.size * t.price if t.size and t.price else 0
            fee = t.fee or 0
            pnl = sell_pnl.get(t.order_id) if t.side == "SELL" else None
            result.append({
                "order_id": t.order_id,
                "product_id": t.product_id,
                "side": t.side,
                "price": t.price,
                "size": t.size,
                "value": value,
                "fee": fee,
                "pnl": round(pnl, 4) if pnl is not None else None,
                "status": t.status,
                "signal_strength": t.signal_strength,
                "strategy": getattr(t, "strategy", "ml") or "ml",
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "filled_at": t.filled_at.isoformat() if t.filled_at else None,
            })
        return result

    @router.get("/trade-frequency")
    def trade_frequency(hours: int = 24):
        since = dt.datetime.utcnow() - dt.timedelta(hours=hours)
        with db.session() as s:
            trades = (
                s.query(Trade)
                .filter(Trade.created_at >= since, Trade.status == "FILLED")
                .order_by(Trade.created_at)
                .all()
            )

        # Adaptive bucket size
        if hours <= 24:
            bucket_secs = 3600        # 1 hour
        elif hours <= 168:
            bucket_secs = 4 * 3600    # 4 hours
        elif hours <= 720:
            bucket_secs = 24 * 3600   # 1 day
        else:
            bucket_secs = 7 * 24 * 3600  # 1 week

        t0 = since.timestamp()
        buckets: dict[int, dict] = {}
        for t in trades:
            ts = t.created_at.timestamp() if t.created_at else t0
            idx = int((ts - t0) // bucket_secs)
            if idx not in buckets:
                buckets[idx] = {"buys": 0, "sells": 0, "ts": t0 + idx * bucket_secs}
            if t.side == "BUY":
                buckets[idx]["buys"] += 1
            else:
                buckets[idx]["sells"] += 1

        result = []
        for idx in sorted(buckets):
            b = buckets[idx]
            result.append({
                "timestamp": b["ts"],
                "buys": b["buys"],
                "sells": b["sells"],
            })
        return {"buckets": result, "bucket_secs": bucket_secs}

    @router.get("/equity")
    def get_equity(hours: int = 168):
        since = dt.datetime.utcnow() - dt.timedelta(hours=hours)
        snapshots = db.get_snapshots_since(since)
        return [
            {
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "total_value": s.total_value_usd,
                "cash": s.cash_usd,
                "holdings": s.holdings_value_usd,
                "drawdown_pct": s.drawdown_pct,
                "eur_usd_rate": s.eur_usd_rate,
            }
            for s in snapshots
        ]

    # ------------------------------------------------------------------
    # Crypto benchmarks
    # ------------------------------------------------------------------

    _BENCHMARK_COINS = ["BTC-USDC", "ETH-USDC", "DOGE-USDC", "SOL-USDC"]
    _benchmark_cache: dict = {"data": None, "hours": None, "ts": 0}

    @router.get("/benchmarks")
    def get_benchmarks(hours: int = 24):
        import time as _time

        now = _time.time()
        cache = _benchmark_cache
        if cache["data"] and cache["hours"] == hours and now - cache["ts"] < 60:
            return cache["data"]

        end = int(now)
        start = end - hours * 3600
        if hours <= 24:
            gran = "FIVE_MINUTE"
        elif hours <= 168:
            gran = "ONE_HOUR"
        elif hours <= 720:
            gran = "SIX_HOUR"
        else:
            gran = "ONE_DAY"

        result: dict = {}
        for pid in _BENCHMARK_COINS:
            label = pid.split("-")[0]
            try:
                candles = client.get_candles(pid, start, end, granularity=gran)
                result[label] = [
                    {"timestamp": c.timestamp, "close": c.close}
                    for c in candles
                ]
            except Exception as exc:
                logger.debug("Benchmark candles failed for %s: %s", pid, exc)
                result[label] = []

        cache["data"] = result
        cache["hours"] = hours
        cache["ts"] = now
        return result

    @router.get("/strategy-performance")
    def strategy_performance():
        closed = db.get_closed_positions()
        buckets: dict[str, list] = {"ml": [], "momentum": [], "external": []}
        wins = {"ml": 0, "momentum": 0, "external": 0}
        losses = {"ml": 0, "momentum": 0, "external": 0}
        sums = {"ml": 0.0, "momentum": 0.0, "external": 0.0}

        all_sorted = sorted(closed, key=lambda p: p.closed_at or dt.datetime.min)
        for p in all_sorted:
            ts = p.closed_at.isoformat() if p.closed_at else None
            pnl = p.pnl or 0.0
            strat = getattr(p, "strategy", "ml") or "ml"
            if strat not in buckets:
                strat = "external"
            buckets[strat].append({
                "timestamp": ts,
                "pnl": round(pnl, 4),
                "product_id": p.product_id,
            })
            sums[strat] += pnl
            if pnl >= 0:
                wins[strat] += 1
            else:
                losses[strat] += 1

        stats = {}
        totals = {}
        for key in buckets:
            total = wins[key] + losses[key]
            totals[key] = round(sums[key], 2)
            stats[key] = {
                "trades": total,
                "wins": wins[key],
                "win_rate": round(wins[key] / total, 3) if total else None,
            }
        totals["combined"] = round(sum(sums.values()), 2)

        return {
            "ml": buckets["ml"],
            "momentum": buckets["momentum"],
            "external": buckets["external"],
            "totals": totals,
            "stats": stats,
        }

    @router.get("/risk")
    def get_risk():
        cap = db.get_capital_record()
        peak = db.get_peak_value()
        snapshots = db.get_snapshots_since(
            dt.datetime.utcnow() - dt.timedelta(hours=1)
        )
        current_value = snapshots[-1].total_value_usd if snapshots else 0

        drawdown_from_peak = (
            (1 - current_value / peak) * 100 if peak > 0 and current_value > 0 else 0
        )
        drawdown_from_initial = 0
        if cap and cap.initial_capital_usd > 0:
            drawdown_from_initial = (1 - current_value / cap.initial_capital_usd) * 100

        return {
            "initial_capital": cap.initial_capital_usd if cap else 0,
            "capital_floor": cap.capital_floor_usd if cap else 0,
            "peak_value": peak,
            "current_value": current_value,
            "drawdown_from_peak_pct": drawdown_from_peak,
            "drawdown_from_initial_pct": drawdown_from_initial,
            "max_loss_pct": cap.max_loss_pct * 100 if cap else 50,
        }

    @router.get("/fx")
    def get_fx():
        if not fx_manager:
            return {"enabled": False}
        fx_s = fx_manager.get_status()
        return {
            "enabled": True,
            "eur_usd_rate": fx_s.eur_usd_rate,
            "portfolio_value_eur": fx_s.portfolio_value_eur,
            "portfolio_value_usd": fx_s.portfolio_value_usd,
            "eurc_balance": fx_s.eurc_balance,
            "usdc_balance": fx_s.usdc_balance,
            "usd_exposure_pct": fx_s.usd_exposure_pct,
        }

    @router.get("/health")
    def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Momentum endpoints
    # ------------------------------------------------------------------

    @router.get("/momentum/status")
    def momentum_status():
        is_active = MomentumScanner.is_flag_active()
        stats = momentum_scanner.get_stats() if momentum_scanner else None
        positions = db.get_open_positions(strategy="momentum")
        watchlist = (momentum_scanner.get_watchlist() if momentum_scanner
                     else MomentumScanner.load_watchlist_file())
        confidence = (momentum_scanner.get_confidence_scores() if momentum_scanner
                      else MomentumScanner.load_confidence_file())

        with db.session() as s:
            total_trades = s.query(Trade).filter(Trade.strategy == "momentum").count()

        if stats:
            coins_scanned = stats.coins_scanned
            candidates_found = stats.candidates_found
            last_scan_time = stats.last_scan_time
        else:
            file_stats = MomentumScanner.load_scan_stats_file()
            coins_scanned = file_stats.get("coins_scanned", 0)
            candidates_found = file_stats.get("candidates_found", 0)
            last_scan_time = file_stats.get("last_scan_time")

        return {
            "enabled": is_active,
            "scanner_running": stats.is_active if stats else False,
            "last_scan_time": last_scan_time,
            "coins_scanned": coins_scanned,
            "candidates_found": candidates_found,
            "open_positions": len(positions),
            "total_trades": total_trades,
            "positions": [
                {
                    "product_id": p.product_id,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "highest_price": p.highest_price,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                }
                for p in positions
            ],
            "watchlist": watchlist,
            "confidence": confidence,
            "untradeable": (
                sorted(momentum_scanner._blacklist) if momentum_scanner
                else MomentumScanner.get_untradeable()
            ),
        }

    @router.post("/momentum/start")
    def momentum_start():
        MomentumScanner.set_flag(True)
        return {"status": "activated"}

    @router.post("/momentum/stop")
    def momentum_stop():
        MomentumScanner.set_flag(False)
        return {"status": "deactivated"}

    @router.post("/momentum/scan")
    def momentum_force_scan():
        if not MomentumScanner.is_flag_active():
            return {"ok": False, "error": "Momentum strategy is disabled"}
        MomentumScanner.request_scan()
        return {"ok": True, "status": "scan triggered"}

    @router.post("/momentum/untradeable/remove/{product_id:path}")
    def remove_untradeable(product_id: str):
        if momentum_scanner:
            momentum_scanner._blacklist.discard(product_id)
            momentum_scanner._save_untradeable()
        else:
            coins = MomentumScanner.get_untradeable()
            if product_id in coins:
                coins.remove(product_id)
                UNTRADEABLE_FILE.write_text(json.dumps(coins), encoding="utf-8")
        return {"ok": True, "product_id": product_id}

    def _close_position_now(product_id: str) -> bool:
        """Sell a position at market and record the trade. Works without a
        live MomentumScanner by using the CoinbaseClient directly."""
        positions = db.get_open_positions(strategy="momentum")
        pos = next((p for p in positions if p.product_id == product_id), None)
        if not pos:
            logger.warning("Close: no open momentum position for %s", product_id)
            return False
        try:
            current_price = client.get_ticker(product_id)
        except Exception as exc:
            logger.error("Close: cannot get price for %s: %s", product_id, exc)
            return False

        size_str = pos.size
        try:
            from decimal import Decimal, ROUND_DOWN
            product = client.get_product(product_id)
            inc = product.base_increment
            d_val = Decimal(str(pos.size))
            d_inc = Decimal(str(inc))
            rounded = float((d_val / d_inc).to_integral_value(rounding=ROUND_DOWN) * d_inc)
            decimals = max(0, -Decimal(str(inc)).normalize().as_tuple().exponent)
            size_str = f"{rounded:.{decimals}f}"
        except Exception:
            size_str = f"{pos.size:.8f}"

        try:
            result = client.place_market_sell(product_id, size_str)
            order_id = (result.order_id if result.order_id else None) or f"api-close-{uuid.uuid4()}"
            sell_value = current_price * pos.size
            taker_fee = sell_value * taker_fee_pct
            gross_pnl = (current_price - pos.entry_price) * pos.size
            net_pnl = gross_pnl - taker_fee
            trade = Trade(
                order_id=order_id,
                product_id=product_id,
                side="SELL",
                size=pos.size,
                price=current_price,
                fee=taker_fee,
                status="FILLED",
                signal_strength=0,
                strategy="momentum",
                notes="momentum exit: manual close (dashboard)",
            )
            db.save_trade(trade)
            db.close_position(product_id, net_pnl)
            _record_decision_yield(
                product_id, pos.entry_price, pos.size,
                pos.opened_at, net_pnl, taker_fee,
                "manual close (dashboard)",
            )
            logger.info(
                "CLOSE %s via dashboard: gross=$%.2f fee=$%.4f net=$%.2f",
                product_id, gross_pnl, taker_fee, net_pnl,
            )
            if fx_manager:
                try:
                    fx_manager.rebalance_to_eurc()
                except Exception:
                    pass
            return True
        except Exception as exc:
            logger.error("Close failed for %s: %s", product_id, exc)
            return False

    @router.post("/momentum/close-all")
    def momentum_close_all():
        if momentum_scanner:
            closed = momentum_scanner.close_all_positions()
            return {"closed": closed}
        positions = db.get_open_positions(strategy="momentum")
        if not positions:
            return {"closed": 0}
        closed = sum(1 for p in positions if _close_position_now(p.product_id))
        return {"closed": closed}

    @router.post("/momentum/close/{product_id}")
    def momentum_close_one(product_id: str):
        if momentum_scanner:
            ok = momentum_scanner.close_position(product_id)
            return {"closed": ok, "product_id": product_id}
        ok = _close_position_now(product_id)
        return {"closed": ok, "product_id": product_id}

    # ------------------------------------------------------------------
    # Generic sell (any strategy)
    # ------------------------------------------------------------------

    @router.post("/sell/{product_id}")
    def sell_position(product_id: str):
        positions = db.get_open_positions()
        pos = next((p for p in positions if p.product_id == product_id), None)
        if not pos:
            return {"closed": False, "error": "No open position", "product_id": product_id}

        strategy = getattr(pos, "strategy", "ml") or "ml"

        try:
            current_price = client.get_ticker(product_id)
        except Exception as exc:
            logger.error("Sell: cannot get price for %s: %s", product_id, exc)
            return {"closed": False, "error": str(exc), "product_id": product_id}

        try:
            from decimal import Decimal, ROUND_DOWN

            product = client.get_product(product_id)
            inc = product.base_increment
            d_val = Decimal(str(pos.size))
            d_inc = Decimal(str(inc))
            rounded = float(
                (d_val / d_inc).to_integral_value(rounding=ROUND_DOWN) * d_inc
            )
            decimals = max(0, -Decimal(str(inc)).normalize().as_tuple().exponent)
            size_str = f"{rounded:.{decimals}f}"
        except Exception:
            size_str = f"{pos.size:.8f}"

        if float(size_str) <= 0:
            db.close_position(product_id, pnl=0)
            return {"closed": True, "product_id": product_id, "note": "dust position cleared"}

        try:
            result = client.place_market_sell(product_id, size_str)
            order_id = (
                (result.order_id if result.order_id else None)
                or f"api-sell-{uuid.uuid4()}"
            )
            sell_value = current_price * pos.size
            taker_fee = sell_value * taker_fee_pct
            gross_pnl = (current_price - pos.entry_price) * pos.size
            net_pnl = gross_pnl - taker_fee
            trade = Trade(
                order_id=order_id,
                product_id=product_id,
                side="SELL",
                size=pos.size,
                price=current_price,
                fee=taker_fee,
                status="FILLED",
                signal_strength=0,
                strategy=strategy,
                notes=f"{strategy} exit: manual sell (dashboard)",
            )
            db.save_trade(trade)
            db.close_position(product_id, net_pnl)
            if strategy == "momentum":
                _record_decision_yield(
                    product_id, pos.entry_price, pos.size,
                    pos.opened_at, net_pnl, taker_fee,
                    "manual sell (dashboard)",
                )
            logger.info(
                "SELL %s via dashboard: gross=$%.2f fee=$%.4f net=$%.2f",
                product_id,
                gross_pnl,
                taker_fee,
                net_pnl,
            )
            if fx_manager:
                try:
                    fx_manager.rebalance_to_eurc()
                except Exception:
                    pass
            return {"closed": True, "product_id": product_id, "pnl": round(net_pnl, 4)}
        except Exception as exc:
            logger.error("Sell failed for %s: %s", product_id, exc)
            return {"closed": False, "error": str(exc), "product_id": product_id}

    # ------------------------------------------------------------------
    # Coin chart (candle data for any product)
    # ------------------------------------------------------------------

    @router.get("/coin-chart/{product_id}")
    def get_coin_chart(product_id: str, hours: int = 1):
        import time as _time

        now = int(_time.time())
        start = now - hours * 3600

        if hours <= 1:
            gran = "ONE_MINUTE"
        elif hours <= 6:
            gran = "FIVE_MINUTE"
        elif hours <= 24:
            gran = "FIVE_MINUTE"
        elif hours <= 168:
            gran = "ONE_HOUR"
        else:
            gran = "SIX_HOUR"

        try:
            candles = client.get_candles(product_id, start, now, granularity=gran)
            return {
                "product_id": product_id,
                "hours": hours,
                "candles": [
                    {
                        "timestamp": c.timestamp,
                        "close": c.close,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "volume": c.volume,
                    }
                    for c in candles
                ],
            }
        except Exception as exc:
            logger.debug("Coin chart failed for %s: %s", product_id, exc)
            return {"product_id": product_id, "hours": hours, "candles": []}

    @router.post("/reset-stop/{product_id}")
    def reset_stop(product_id: str):
        """Set stop-loss to current price minus 1%."""
        positions = db.get_open_positions()
        pos = next((p for p in positions if p.product_id == product_id), None)
        if not pos:
            return {"ok": False, "error": "No open position", "product_id": product_id}
        try:
            price = client.get_ticker(product_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "product_id": product_id}
        new_sl = round(price * 0.99, 10)
        db.update_position(product_id, stop_loss=new_sl, highest_price=price)
        logger.info("RESET-STOP %s: new SL=$%.10f (price=$%.10f)", product_id, new_sl, price)
        return {"ok": True, "product_id": product_id, "stop_loss": new_sl, "price": price}

    # ------------------------------------------------------------------
    # Blocklist
    # ------------------------------------------------------------------

    @router.get("/blocklist")
    def get_blocklist():
        coins = db.get_blocked_coins()
        return [
            {
                "product_id": c.product_id,
                "blocked_at": c.blocked_at.isoformat() if c.blocked_at else None,
                "reason": c.reason,
            }
            for c in coins
        ]

    @router.post("/block/{product_id}")
    def block_coin(product_id: str):
        added = db.block_coin(product_id, reason="manual")
        logger.info("BLOCKLIST: %s %s", "blocked" if added else "already blocked", product_id)
        return {"ok": True, "product_id": product_id, "added": added}

    @router.post("/unblock/{product_id}")
    def unblock_coin(product_id: str):
        removed = db.unblock_coin(product_id)
        logger.info("BLOCKLIST: %s %s", "unblocked" if removed else "not found", product_id)
        return {"ok": True, "product_id": product_id, "removed": removed}

    # ------------------------------------------------------------------
    # Position reconciliation (sync DB with Coinbase)
    # ------------------------------------------------------------------

    _SKIP_CURRENCIES = (
        {"USD"}
        | CoinbaseClient._USD_STABLECOINS
        | CoinbaseClient._EUR_STABLECOINS
    )

    def _infer_strategy(product_id: str) -> str:
        """Infer the strategy for a coin from its recent trades or closed positions."""
        trades = db.get_recent_trades_for(
            product_id, dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
        )
        for t in trades:
            strat = getattr(t, "strategy", None)
            if strat in ("ml", "momentum"):
                return strat
        closed = db.get_closed_positions()
        for p in reversed(closed):
            if p.product_id == product_id:
                strat = getattr(p, "strategy", None)
                if strat in ("ml", "momentum"):
                    return strat
        return "external"

    _DUST_USD = 1.0

    def _build_reconcile_diff() -> dict:
        accounts = client.get_accounts()
        coinbase_holdings: dict[str, float] = {}
        for acct in accounts:
            if acct.currency in _SKIP_CURRENCIES:
                continue
            if acct.total <= 0:
                continue
            product_id = f"{acct.currency}-USDC"
            coinbase_holdings[product_id] = acct.total

        db_positions = db.get_open_positions()
        db_map: dict[str, Position] = {p.product_id: p for p in db_positions}

        phantoms = []
        orphans = []
        size_updates = []
        matched = []

        for pid, pos in db_map.items():
            cb_size = coinbase_holdings.get(pid)
            if cb_size is None or cb_size <= 0:
                phantoms.append({
                    "product_id": pid,
                    "db_size": pos.size,
                    "coinbase_size": 0,
                    "strategy": getattr(pos, "strategy", "ml") or "ml",
                })
            else:
                rel_diff = abs(cb_size - pos.size) / max(pos.size, 1e-12)
                if rel_diff > 0.001:
                    size_updates.append({
                        "product_id": pid,
                        "db_size": pos.size,
                        "coinbase_size": cb_size,
                    })
                else:
                    matched.append(pid)

        for pid, cb_size in coinbase_holdings.items():
            if pid not in db_map:
                try:
                    price = client.get_ticker(pid)
                except Exception:
                    price = 0
                value = cb_size * price
                if value < _DUST_USD:
                    continue
                orphans.append({
                    "product_id": pid,
                    "coinbase_size": cb_size,
                    "current_price": price,
                    "value_usd": round(value, 2),
                })

        return {
            "phantoms": phantoms,
            "orphans": orphans,
            "size_updates": size_updates,
            "matched": matched,
        }

    @router.get("/reconcile")
    def reconcile_preview():
        return _build_reconcile_diff()

    @router.post("/reconcile")
    def reconcile_apply():
        diff = _build_reconcile_diff()

        closed = []
        for p in diff["phantoms"]:
            pid = p["product_id"]
            db.close_position(pid, pnl=0)
            logger.info("RECONCILE: closed phantom position %s (no Coinbase balance)", pid)
            closed.append(pid)

        imported = []
        for o in diff["orphans"]:
            pid = o["product_id"]
            price = o["current_price"]
            size = o["coinbase_size"]
            if price <= 0 or size <= 0:
                continue
            strat = _infer_strategy(pid)
            sl = price * 0.92
            tp = price * 1.06
            pos = Position(
                product_id=pid,
                side="LONG",
                entry_price=price,
                size=size,
                stop_loss=sl,
                take_profit=tp,
                highest_price=price,
                strategy=strat,
            )
            db.save_position(pos)
            logger.info(
                "RECONCILE: imported orphan %s [%s] (size=%.8f, price=$%.4f, SL=$%.4f, TP=$%.4f)",
                pid, strat, size, price, sl, tp,
            )
            imported.append(pid)

        updated = []
        for u in diff["size_updates"]:
            pid = u["product_id"]
            db.update_position(pid, size=u["coinbase_size"])
            logger.info(
                "RECONCILE: updated size for %s (%.8f -> %.8f)",
                pid, u["db_size"], u["coinbase_size"],
            )
            updated.append(pid)

        return {
            "phantoms_closed": closed,
            "orphans_imported": imported,
            "sizes_updated": updated,
            "matched": diff["matched"],
        }

    # ------------------------------------------------------------------
    # Automatic periodic position sync
    # ------------------------------------------------------------------

    _sync_state = {
        "last_sync": None,
        "last_result": None,
        "next_sync": None,
        "running": False,
    }

    def _auto_sync_once() -> dict | None:
        """Run one reconciliation cycle silently.  Returns the result dict or
        None if nothing needed fixing."""
        try:
            diff = _build_reconcile_diff()
        except Exception as exc:
            logger.error("Auto-sync: failed to build diff: %s", exc)
            return None

        total_changes = len(diff["phantoms"]) + len(diff["orphans"]) + len(diff["size_updates"])
        if total_changes == 0:
            return {"changes": 0, "matched": len(diff["matched"])}

        closed = []
        for p in diff["phantoms"]:
            pid = p["product_id"]
            db.close_position(pid, pnl=0)
            logger.info("AUTO-SYNC: closed phantom position %s", pid)
            closed.append(pid)

        imported = []
        for o in diff["orphans"]:
            pid = o["product_id"]
            price = o["current_price"]
            size = o["coinbase_size"]
            if price <= 0 or size <= 0:
                continue
            strat = _infer_strategy(pid)
            sl = price * 0.92
            tp = price * 1.06
            pos = Position(
                product_id=pid,
                side="LONG",
                entry_price=price,
                size=size,
                stop_loss=sl,
                take_profit=tp,
                highest_price=price,
                strategy=strat,
            )
            db.save_position(pos)
            logger.info("AUTO-SYNC: imported orphan %s [%s] (%.8f @ $%.4f, SL=$%.4f)", pid, strat, size, price, sl)
            imported.append(pid)

        updated = []
        for u in diff["size_updates"]:
            pid = u["product_id"]
            db.update_position(pid, size=u["coinbase_size"])
            logger.info("AUTO-SYNC: size update %s (%.8f → %.8f)", pid, u["db_size"], u["coinbase_size"])
            updated.append(pid)

        return {
            "changes": len(closed) + len(imported) + len(updated),
            "phantoms_closed": closed,
            "orphans_imported": imported,
            "sizes_updated": updated,
            "matched": len(diff["matched"]),
        }

    def _fix_missing_stops() -> None:
        """Set default stop/target on positions that lack them, and fix strategy labels."""
        for pos in db.get_open_positions():
            updates: dict = {}

            if not pos.stop_loss or pos.stop_loss <= 0:
                if pos.entry_price and pos.entry_price > 0:
                    try:
                        current_price = client.get_ticker(pos.product_id)
                    except Exception:
                        current_price = pos.entry_price
                    ref_price = max(pos.entry_price, current_price)
                    updates["stop_loss"] = current_price * 0.92
                    updates["take_profit"] = ref_price * 1.06
                    updates["highest_price"] = current_price
                    logger.info(
                        "AUTO-FIX: set stops on %s (SL=$%.6f, TP=$%.6f)",
                        pos.product_id, updates["stop_loss"], updates["take_profit"],
                    )

            strat = getattr(pos, "strategy", None) or ""
            if strat in ("external", "manual", ""):
                inferred = _infer_strategy(pos.product_id)
                if inferred != "external":
                    updates["strategy"] = inferred
                    logger.info("AUTO-FIX: relabeled %s strategy %s → %s", pos.product_id, strat, inferred)

            if updates:
                db.update_position(pos.product_id, **updates)

    def _sync_loop() -> None:
        """Background thread that reconciles positions periodically."""
        _sync_state["running"] = True
        # Fix positions missing stops, then sync
        try:
            _fix_missing_stops()
        except Exception as exc:
            logger.error("Auto-fix stops error: %s", exc)
        try:
            result = _auto_sync_once()
            _sync_state["last_sync"] = dt.datetime.utcnow().isoformat()
            _sync_state["last_result"] = result
            if result and result.get("changes", 0) > 0:
                logger.info("AUTO-SYNC (startup): applied %d change(s)", result["changes"])
        except Exception as exc:
            logger.error("Auto-sync startup error: %s", exc)
        while _sync_state["running"]:
            _sync_state["next_sync"] = dt.datetime.utcnow() + dt.timedelta(seconds=sync_interval_sec)
            for _ in range(sync_interval_sec):
                if not _sync_state["running"]:
                    return
                time.sleep(1)
            try:
                result = _auto_sync_once()
                _sync_state["last_sync"] = dt.datetime.utcnow().isoformat()
                _sync_state["last_result"] = result
                if result and result.get("changes", 0) > 0:
                    logger.info("AUTO-SYNC: applied %d change(s)", result["changes"])
            except Exception as exc:
                logger.error("Auto-sync error: %s", exc)

    if sync_interval_sec > 0:
        _sync_thread = threading.Thread(target=_sync_loop, daemon=True, name="position-auto-sync")
        _sync_thread.start()

    @router.get("/sync-status")
    def get_sync_status():
        return {
            "enabled": sync_interval_sec > 0,
            "interval_sec": sync_interval_sec,
            "last_sync": _sync_state["last_sync"],
            "last_result": _sync_state["last_result"],
            "next_sync": _sync_state["next_sync"].isoformat() if _sync_state["next_sync"] else None,
        }

    # ------------------------------------------------------------------
    # Trade / order history reconciliation
    # ------------------------------------------------------------------

    def _build_trade_diff() -> dict:
        """Compare Coinbase filled orders against the local trades table."""
        cb_orders = client.list_filled_orders(limit=200)

        existing_ids: set[str] = set()
        with db.session() as s:
            rows = s.query(Trade.order_id).all()
            existing_ids = {r[0] for r in rows}

        missing: list[dict] = []
        already_synced: list[str] = []

        for o in cb_orders:
            oid = o.get("order_id", "")
            if not oid:
                continue
            if oid in existing_ids:
                already_synced.append(oid)
                continue

            pid = o.get("product_id", "")
            side = o.get("side", "UNKNOWN")
            filled_size = float(o.get("filled_size") or 0)
            avg_price = float(o.get("average_filled_price") or 0)
            total_fees = float(o.get("total_fees") or 0)
            created = o.get("created_time", "")
            fill_time = o.get("last_fill_time", "")

            if filled_size <= 0 or avg_price <= 0:
                continue

            missing.append({
                "order_id": oid,
                "product_id": pid,
                "side": side,
                "size": filled_size,
                "price": avg_price,
                "fee": total_fees,
                "value": round(filled_size * avg_price, 4),
                "created_time": created,
                "fill_time": fill_time,
            })

        return {
            "missing": missing,
            "already_synced": len(already_synced),
            "coinbase_total": len(cb_orders),
        }

    @router.get("/reconcile-trades")
    def reconcile_trades_preview():
        return _build_trade_diff()

    @router.post("/reconcile-trades")
    def reconcile_trades_apply():
        diff = _build_trade_diff()

        imported = []
        for m in diff["missing"]:
            created_dt = None
            if m["created_time"]:
                try:
                    created_dt = dt.datetime.fromisoformat(
                        m["created_time"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            filled_dt = None
            if m["fill_time"]:
                try:
                    filled_dt = dt.datetime.fromisoformat(
                        m["fill_time"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            trade = Trade(
                order_id=m["order_id"],
                product_id=m["product_id"],
                side=m["side"],
                size=m["size"],
                price=m["price"],
                fee=m["fee"],
                status="FILLED",
                signal_strength=0,
                strategy="external",
                created_at=created_dt,
                filled_at=filled_dt,
                notes="imported from Coinbase (not placed by Trade-ng)",
            )
            try:
                db.save_trade(trade)
                imported.append(m["order_id"])
                logger.info(
                    "TRADE SYNC: imported %s %s %s (%.8f @ $%.4f, fee=$%.4f)",
                    m["side"], m["product_id"], m["order_id"],
                    m["size"], m["price"], m["fee"],
                )
            except Exception as exc:
                logger.warning("TRADE SYNC: skip %s: %s", m["order_id"], exc)

        return {
            "imported": len(imported),
            "imported_ids": imported,
            "already_synced": diff["already_synced"],
            "coinbase_total": diff["coinbase_total"],
        }

    return router
