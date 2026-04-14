"""Order execution engine: converts signals into orders with full audit trail."""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid

from src.data.coinbase_client import CoinbaseClient, OrderResult
from src.data.database import Database, Position, Trade
from src.models.ensemble import Signal, TradeSignal
from src.trading.fx_manager import FXManager
from src.trading.risk import CooldownTracker, RiskManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes trades based on validated signals."""

    def __init__(
        self,
        client: CoinbaseClient,
        db: Database,
        risk: RiskManager,
        order_timeout_sec: int = 60,
        maker_fee_pct: float = 0.006,
        taker_fee_pct: float = 0.012,
        fx_manager: FXManager | None = None,
        cooldown: CooldownTracker | None = None,
    ):
        self.client = client
        self.db = db
        self.risk = risk
        self.timeout = order_timeout_sec
        self.maker_fee = maker_fee_pct
        self.taker_fee = taker_fee_pct
        self.fx = fx_manager
        self.cooldown = cooldown
        self._product_cache: dict = {}

    def execute_signal(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        current_price: float,
        atr: float,
        coin_universe: list[str],
    ) -> OrderResult | None:
        """Execute a trade signal after validation through risk manager."""
        product_id = signal.product_id

        if signal.signal == Signal.HOLD:
            return None

        # Determine side
        side = "BUY" if signal.signal == Signal.BUY else "SELL"

        # For SELL signals, check if we have a position to sell
        if side == "SELL":
            position = self.db.get_position(product_id)
            if not position:
                return None
            return self._execute_sell(position, current_price, signal)

        # Skip if we already hold this coin
        existing = self.db.get_position(product_id)
        if existing and existing.is_open:
            return None

        # Skip if coin is on the persistent blocklist
        if self.db.is_coin_blocked(product_id):
            logger.info("BUY blocked %s: on blocklist", product_id)
            return None

        # Skip if coin is on cooldown (recently stopped out)
        if self.cooldown and self.cooldown.is_blocked(product_id):
            remaining = self.cooldown.remaining_minutes(product_id)
            logger.info("BUY blocked %s: on cooldown (%.0fmin remaining)", product_id, remaining)
            return None

        # Skip if coin has repeated stop-losses recently
        recent_stops = self._count_recent_stop_losses(product_id)
        if recent_stops >= 2:
            logger.info("BUY blocked %s: %d stop-losses in last 4h — avoiding repeat losses", product_id, recent_stops)
            return None

        size_usd = self.risk.calculate_position_size(
            portfolio_value=portfolio_value,
            signal_strength=signal.ensemble_score,
            atr=atr,
            price=current_price,
        )

        if size_usd <= 0:
            return None

        current_positions = [
            p.to_dict() if hasattr(p, "to_dict") else {"product_id": p.product_id, "is_open": p.is_open, "value_usd": p.size * current_price}
            for p in self.db.get_open_positions()
        ]

        check = self.risk.validate_trade(
            product_id=product_id,
            side=side,
            size_usd=size_usd,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            coin_universe=coin_universe,
        )

        if not check.allowed:
            logger.info("Trade rejected: %s %s — %s", side, product_id, check.reason)
            return None

        size_usd = check.max_size_usd
        return self._execute_buy(product_id, size_usd, current_price, atr, signal)

    def _count_recent_stop_losses(self, product_id: str, hours: int = 4) -> int:
        """Count stop-loss sells for this coin in the last N hours."""
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
        trades = self.db.get_recent_trades_for(product_id, since)
        count = 0
        for t in trades:
            if t.side == "SELL" and t.notes and "stop-loss" in t.notes.lower():
                count += 1
        return count

    @staticmethod
    def _increment_decimals(increment: float) -> int:
        """Number of decimal places in an increment."""
        from decimal import Decimal
        d = Decimal(str(increment)).normalize()
        return max(0, -d.as_tuple().exponent)

    @staticmethod
    def _round_to_increment(value: float, increment: float) -> float:
        """Round a value down to the nearest valid increment."""
        if increment <= 0:
            return value
        from decimal import Decimal, ROUND_DOWN
        d_val = Decimal(str(value))
        d_inc = Decimal(str(increment))
        return float((d_val / d_inc).to_integral_value(rounding=ROUND_DOWN) * d_inc)

    def _get_product(self, product_id: str):
        if product_id not in self._product_cache:
            self._product_cache[product_id] = self.client.get_product(product_id)
        return self._product_cache[product_id]

    def _format_size(self, size: float, product_id: str) -> str:
        """Format order size respecting product's base_increment."""
        try:
            product = self._get_product(product_id)
            inc = product.base_increment
            size = self._round_to_increment(size, inc)
            decimals = self._increment_decimals(inc)
            return f"{size:.{decimals}f}"
        except Exception:
            return f"{size:.8f}"

    def _format_price(self, price: float, product_id: str | None = None) -> str:
        """Format price respecting product's quote_increment."""
        if product_id:
            try:
                product = self._get_product(product_id)
                inc = product.quote_increment
                price = self._round_to_increment(price, inc)
                decimals = self._increment_decimals(inc)
                return f"{price:.{decimals}f}"
            except Exception:
                pass
        if price >= 1.0:
            return f"{price:.2f}"
        elif price >= 0.001:
            return f"{price:.6f}"
        else:
            return f"{price:.10f}"

    def _execute_buy(
        self,
        product_id: str,
        size_usd: float,
        price: float,
        atr: float,
        signal: TradeSignal,
        strategy: str = "ml",
    ) -> OrderResult | None:
        """Place a limit buy order, reserving funds for fees."""
        if price <= 0:
            return None

        if self.fx:
            if not self.fx.ensure_usdc_for_trade(size_usd):
                logger.info("FX: could not secure USDC for %s ($%.2f)", product_id, size_usd)
                return None
            actual_usdc = self.client.get_usd_balance()
            if actual_usdc < size_usd:
                if actual_usdc < 5.0:
                    logger.info("FX: USDC still insufficient for %s ($%.2f available)", product_id, actual_usdc)
                    return None
                size_usd = actual_usdc

        # Reserve funds for maker fee (buy) + taker fee (worst-case sell)
        round_trip_fee = self.maker_fee + self.taker_fee
        effective_usd = size_usd / (1 + round_trip_fee)
        estimated_fee = size_usd - effective_usd

        size_base = effective_usd / price
        size_str = self._format_size(size_base, product_id)
        price_str = self._format_price(price, product_id)

        logger.info(
            "Placing BUY %s: %s @ $%s ($%.2f, fee reserve $%.2f) signal=%.3f",
            product_id,
            size_str,
            price_str,
            effective_usd,
            estimated_fee,
            signal.ensemble_score,
        )

        try:
            result = self.client.place_limit_buy(
                product_id=product_id,
                size=size_str,
                price=price_str,
            )
        except Exception as e:
            logger.error("Order placement failed for %s: %s", product_id, e)
            return None

        if result.status == "FAILED":
            logger.warning("Order rejected by exchange for %s: %s", product_id, result.raw)
            return None

        buy_fee = effective_usd * self.maker_fee
        order_id = result.order_id or f"local-{uuid.uuid4()}"
        trade = Trade(
            order_id=order_id,
            product_id=product_id,
            side="BUY",
            size=size_base,
            price=price,
            fee=buy_fee,
            status=result.status,
            signal_strength=signal.ensemble_score,
            strategy=strategy,
            notes=f"xgb={signal.xgb_prob:.3f} lstm={signal.lstm_pred:.5f}",
        )
        self.db.save_trade(trade)

        if result.status == "PENDING":
            filled, fill_data = self._wait_for_fill(result.order_id)
            if filled:
                actual_fee = self._extract_fee(fill_data, buy_fee)
                self.db.update_trade(order_id, fee=actual_fee)

                stop_loss = self.risk.calculate_stop_loss(price, atr)
                take_profit = self.risk.calculate_take_profit(price, atr)
                position = Position(
                    product_id=product_id,
                    side="LONG",
                    entry_price=price,
                    size=size_base,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    highest_price=price,
                    strategy=strategy,
                )
                self.db.save_position(position)
                self.db.update_trade(order_id, status="FILLED", filled_at=dt.datetime.now(dt.timezone.utc))
                logger.info(
                    "BUY FILLED %s: %s @ $%s  fee=$%.4f  SL=$%s  TP=$%s",
                    product_id,
                    size_str,
                    price_str,
                    actual_fee,
                    self._format_price(stop_loss, product_id),
                    self._format_price(take_profit, product_id),
                )
            else:
                self.client.cancel_orders([result.order_id])
                self.db.update_trade(order_id, status="CANCELLED")
                logger.info("BUY order timed out and cancelled: %s", product_id)
                return None

        return result

    def _execute_sell(
        self,
        position: Position,
        current_price: float,
        signal: TradeSignal,
        strategy: str = "ml",
    ) -> OrderResult | None:
        """Place a limit sell order to close a position."""
        product_id = position.product_id

        sell_value = current_price * position.size
        sell_fee_est = sell_value * self.maker_fee
        gross_pnl = (current_price - position.entry_price) * position.size
        net_pnl = gross_pnl - sell_fee_est

        size_str = self._format_size(position.size, product_id)
        price_str = self._format_price(current_price, product_id)

        if float(size_str) <= 0:
            logger.warning("SELL skipped %s: size rounds to 0 (dust position %.10f)", product_id, position.size)
            self.db.close_position(product_id, pnl=0)
            return None

        logger.info(
            "Placing SELL %s: %s @ $%s  gross=$%.2f  fee~$%.4f  net=$%.2f",
            product_id,
            size_str,
            price_str,
            gross_pnl,
            sell_fee_est,
            net_pnl,
        )

        try:
            result = self.client.place_limit_sell(
                product_id=product_id,
                size=size_str,
                price=price_str,
            )
        except Exception as e:
            logger.error("Sell order failed for %s: %s", product_id, e)
            return None

        if result.status == "FAILED":
            logger.warning("Sell order rejected by exchange for %s: %s", product_id, result.raw)
            return None

        order_id = result.order_id or f"local-{uuid.uuid4()}"
        trade = Trade(
            order_id=order_id,
            product_id=product_id,
            side="SELL",
            size=position.size,
            price=current_price,
            fee=sell_fee_est,
            status=result.status,
            signal_strength=signal.ensemble_score,
            strategy=strategy,
        )
        self.db.save_trade(trade)

        if result.status == "PENDING":
            filled, fill_data = self._wait_for_fill(result.order_id)
            if filled:
                actual_fee = self._extract_fee(fill_data, sell_fee_est)
                self.db.update_trade(order_id, fee=actual_fee)
                pnl = gross_pnl - actual_fee
                self.db.close_position(product_id, pnl)
                self.db.update_trade(order_id, status="FILLED", filled_at=dt.datetime.now(dt.timezone.utc))
                logger.info("SELL FILLED %s: gross=$%.2f  fee=$%.4f  net P&L=$%.2f", product_id, gross_pnl, actual_fee, pnl)
                if self.fx:
                    self.fx.rebalance_to_eurc()
            else:
                self.client.cancel_orders([result.order_id])
                self.db.update_trade(order_id, status="CANCELLED")
                logger.info("SELL order timed out: %s", product_id)
                return None

        return result

    def emergency_liquidate(self) -> None:
        """Emergency: sell all positions using market orders."""
        logger.critical("EMERGENCY LIQUIDATION — selling all positions")

        # Cancel all open orders first
        open_orders = self.client.list_open_orders()
        if open_orders:
            order_ids = [o.get("order_id", "") for o in open_orders if o.get("order_id")]
            self.client.cancel_orders(order_ids)
            logger.info("Cancelled %d open orders", len(order_ids))

        # Market-sell all positions
        positions = self.db.get_open_positions()
        for pos in positions:
            try:
                size_str = self._format_size(pos.size, pos.product_id)
                if float(size_str) <= 0:
                    logger.warning("Emergency skip %s: dust position", pos.product_id)
                    self.db.close_position(pos.product_id, pnl=0)
                    continue
                result = self.client.place_market_sell(
                    product_id=pos.product_id,
                    size=size_str,
                )
                trade = Trade(
                    order_id=result.order_id,
                    product_id=pos.product_id,
                    side="SELL",
                    size=pos.size,
                    price=0,
                    status="EMERGENCY_SELL",
                    notes="Emergency liquidation",
                )
                self.db.save_trade(trade)
                self.db.close_position(pos.product_id, pnl=0)
                logger.info("Emergency sold %s: %.6f", pos.product_id, pos.size)
            except Exception as e:
                logger.error("Failed to emergency sell %s: %s", pos.product_id, e)

    def check_stops(self, prices: dict[str, float]) -> list[str]:
        """Check all open positions for stop-loss or take-profit triggers."""
        closed = []
        positions = self.db.get_open_positions()

        for pos in positions:
            current_price = prices.get(pos.product_id, 0)
            if current_price <= 0:
                continue

            if current_price > (pos.highest_price or 0):
                self.db.update_position(
                    pos.product_id, highest_price=current_price
                )
                if pos.stop_loss and pos.entry_price:
                    atr_dist = pos.entry_price - pos.stop_loss
                    gain_pct = (current_price / pos.entry_price - 1) if pos.entry_price > 0 else 0

                    # Tighten trailing stop as profit grows
                    if gain_pct >= 0.03:
                        trail_dist = atr_dist * 0.35
                    elif gain_pct >= 0.02:
                        trail_dist = atr_dist * 0.50
                    elif gain_pct >= 0.01:
                        trail_dist = atr_dist * 0.70
                    else:
                        trail_dist = atr_dist

                    new_stop = current_price - trail_dist
                    if new_stop > pos.stop_loss:
                        self.db.update_position(pos.product_id, stop_loss=new_stop)

            if pos.stop_loss and self.risk.check_stop_loss(current_price, pos.stop_loss):
                logger.warning(
                    "STOP-LOSS triggered %s: $%s <= $%s",
                    pos.product_id,
                    self._format_price(current_price, pos.product_id),
                    self._format_price(pos.stop_loss, pos.product_id),
                )
                try:
                    size_str = self._format_size(pos.size, pos.product_id)
                    if float(size_str) <= 0:
                        logger.warning("Stop-loss skipped %s: dust position (%.10f)", pos.product_id, pos.size)
                        self.db.close_position(pos.product_id, pnl=0)
                        closed.append(pos.product_id)
                        continue
                    result = self.client.place_market_sell(pos.product_id, size_str)
                    sell_value = current_price * pos.size
                    taker_fee = sell_value * self.taker_fee
                    gross_pnl = (current_price - pos.entry_price) * pos.size
                    net_pnl = gross_pnl - taker_fee
                    order_id = result.order_id or f"sl-{uuid.uuid4()}"
                    trade = Trade(
                        order_id=order_id,
                        product_id=pos.product_id,
                        side="SELL",
                        size=pos.size,
                        price=current_price,
                        fee=taker_fee,
                        status="FILLED",
                        signal_strength=0,
                        strategy=getattr(pos, "strategy", "ml") or "ml",
                        notes="stop-loss",
                    )
                    self.db.save_trade(trade)
                    self.db.close_position(pos.product_id, net_pnl)
                    closed.append(pos.product_id)
                    logger.info("Stop-loss SOLD %s: gross=$%.2f fee=$%.4f net=$%.2f", pos.product_id, gross_pnl, taker_fee, net_pnl)
                    if self.cooldown:
                        self.cooldown.add(pos.product_id, "stop-loss")
                    if self.fx:
                        self.fx.rebalance_to_eurc()
                except Exception as e:
                    logger.error("Stop-loss sell failed %s: %s", pos.product_id, e)

            elif pos.take_profit and self.risk.check_take_profit(current_price, pos.take_profit):
                logger.info(
                    "TAKE-PROFIT triggered %s: $%s >= $%s",
                    pos.product_id,
                    self._format_price(current_price, pos.product_id),
                    self._format_price(pos.take_profit, pos.product_id),
                )
                try:
                    size_str = self._format_size(pos.size, pos.product_id)
                    if float(size_str) <= 0:
                        logger.warning("Take-profit skipped %s: dust position (%.10f)", pos.product_id, pos.size)
                        self.db.close_position(pos.product_id, pnl=0)
                        closed.append(pos.product_id)
                        continue
                    price_str = self._format_price(current_price, pos.product_id)
                    result = self.client.place_limit_sell(pos.product_id, size_str, price_str)
                    sell_value = current_price * pos.size
                    maker_fee = sell_value * self.maker_fee
                    gross_pnl = (current_price - pos.entry_price) * pos.size
                    net_pnl = gross_pnl - maker_fee
                    order_id = result.order_id or f"tp-{uuid.uuid4()}"
                    trade = Trade(
                        order_id=order_id,
                        product_id=pos.product_id,
                        side="SELL",
                        size=pos.size,
                        price=current_price,
                        fee=maker_fee,
                        status="FILLED",
                        signal_strength=0,
                        strategy=getattr(pos, "strategy", "ml") or "ml",
                        notes="take-profit",
                    )
                    self.db.save_trade(trade)
                    self.db.close_position(pos.product_id, net_pnl)
                    closed.append(pos.product_id)
                    logger.info("Take-profit SOLD %s: gross=$%.2f fee=$%.4f net=$%.2f", pos.product_id, gross_pnl, maker_fee, net_pnl)
                    if self.fx:
                        self.fx.rebalance_to_eurc()
                except Exception as e:
                    logger.error("Take-profit sell failed %s: %s", pos.product_id, e)

        return closed

    def _wait_for_fill(self, order_id: str) -> tuple[bool, dict]:
        """Poll order status until filled or timeout. Returns (filled, order_data)."""
        if not order_id:
            logger.info("No order ID returned — treating as immediate fill")
            return True, {}

        not_found_count = 0
        time.sleep(0.5)  # brief pause for exchange settlement

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                order = self.client.get_order(order_id)
                order_data = order.get("order", {})
                status = order_data.get("status", "")
                if status in ("FILLED", "COMPLETED"):
                    return True, order_data
                if status in ("CANCELLED", "FAILED", "EXPIRED"):
                    return False, order_data
                not_found_count = 0
            except Exception as exc:
                err_str = str(exc).lower()
                if "not found" in err_str or "404" in err_str:
                    not_found_count += 1
                    if not_found_count >= 3:
                        logger.info(
                            "Order %s not found after %d attempts — likely filled instantly",
                            order_id, not_found_count,
                        )
                        return True, {}
                else:
                    logger.error("Order check failed %s: %s", order_id, exc)
                    return False, {}
            time.sleep(2)
        return False, {}

    @staticmethod
    def _extract_fee(fill_data: dict, estimated_fee: float) -> float:
        """Extract the actual fee from a filled order response, or fall back to estimate."""
        try:
            fee_str = fill_data.get("total_fees", "")
            if fee_str:
                return float(fee_str)
        except (ValueError, TypeError):
            pass
        return estimated_fee
