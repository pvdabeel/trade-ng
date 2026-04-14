"""6-layer capital protection and risk management.

Layer 1: Spot-only trading (structural — enforced at client level)
Layer 2: Initial capital floor with hard stop at 50% loss
Layer 3: Progressive drawdown throttling
Layer 4: Per-trade and daily/hourly limits
Layer 5: Pre-trade validation gate
Layer 6: Continuous portfolio monitoring (independent watchdog)
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.data.coinbase_client import CoinbaseClient
from src.data.database import Database, PortfolioSnapshot

logger = logging.getLogger(__name__)


class TradingState(Enum):
    NORMAL = "NORMAL"
    THROTTLED_50 = "THROTTLED_50"   # 10-20% drawdown
    THROTTLED_25 = "THROTTLED_25"   # 20-30% drawdown
    SELL_ONLY = "SELL_ONLY"         # 30-40% drawdown
    HALTED_DAILY = "HALTED_DAILY"   # daily loss limit hit
    HALTED_HOURLY = "HALTED_HOURLY" # hourly loss limit hit
    EMERGENCY = "EMERGENCY"         # capital floor breached
    SHUTDOWN = "SHUTDOWN"           # permanently stopped


@dataclass
class RiskConfig:
    max_loss_pct: float = 0.50
    max_position_pct: float = 0.05
    max_concentration_pct: float = 0.20
    max_open_positions: int = 10
    daily_loss_limit_pct: float = 0.03
    hourly_loss_limit_pct: float = 0.015
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    watchdog_interval_sec: int = 30


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""
    max_size_usd: float = 0.0
    position_scale: float = 1.0  # 1.0 = normal, 0.5 = throttled, etc.


class CooldownTracker:
    """Cross-strategy cooldown: prevents re-entry after a stop-loss exit."""

    def __init__(self, default_minutes: int = 30):
        self._default_minutes = default_minutes
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    def seed_from_db(self, db) -> None:
        """Populate cooldowns from recent stop-loss trades in the DB."""
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=self._default_minutes)
        try:
            trades = db.get_recent_trades(limit=200)
            for t in trades:
                if t.side != "SELL":
                    continue
                notes = (t.notes or "").lower()
                if "stop-loss" not in notes and "stop_loss" not in notes:
                    continue
                if t.filled_at and t.filled_at.replace(tzinfo=dt.timezone.utc) > since:
                    remaining = self._default_minutes - (
                        dt.datetime.now(dt.timezone.utc) - t.filled_at.replace(tzinfo=dt.timezone.utc)
                    ).total_seconds() / 60
                    if remaining > 1:
                        with self._lock:
                            self._cooldowns[t.product_id] = time.time() + remaining * 60
                        logger.info(
                            "Cooldown: restored %s (%.0fmin remaining from recent stop-loss)",
                            t.product_id, remaining,
                        )
        except Exception as exc:
            logger.warning("Failed to seed cooldowns from DB: %s", exc)

    def add(self, product_id: str, reason: str, minutes: int | None = None) -> None:
        cd_min = minutes if minutes is not None else self._default_minutes
        with self._lock:
            self._cooldowns[product_id] = time.time() + cd_min * 60
        logger.info("Cooldown: %s blocked for %dmin (%s)", product_id, cd_min, reason)

    def is_blocked(self, product_id: str) -> bool:
        with self._lock:
            expiry = self._cooldowns.get(product_id)
            if expiry is None:
                return False
            if time.time() < expiry:
                return True
            del self._cooldowns[product_id]
            return False

    def remaining_minutes(self, product_id: str) -> float:
        with self._lock:
            expiry = self._cooldowns.get(product_id)
            if expiry is None:
                return 0.0
            rem = (expiry - time.time()) / 60
            return max(0.0, rem)


class RiskManager:
    """Central risk manager implementing all 6 protection layers."""

    def __init__(
        self,
        config: RiskConfig,
        client: CoinbaseClient,
        db: Database,
        on_emergency: Callable[[], None] | None = None,
        fx_manager: object | None = None,
    ):
        self.config = config
        self.client = client
        self.db = db
        self._on_emergency = on_emergency
        self._fx = fx_manager

        self._state = TradingState.NORMAL
        self._initial_capital: float = 0.0
        self._capital_floor: float = 0.0
        self._peak_value: float = 0.0

        self._daily_start_value: float = 0.0
        self._daily_start_time: dt.datetime = dt.datetime.now(dt.timezone.utc)
        self._hourly_start_value: float = 0.0
        self._hourly_start_time: dt.datetime = dt.datetime.now(dt.timezone.utc)

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_running = False

        self._lock = threading.Lock()

    @property
    def state(self) -> TradingState:
        return self._state

    # ------------------------------------------------------------------
    # Layer 2: Initial capital floor
    # ------------------------------------------------------------------

    def initialize_capital(self, prices: dict[str, float] | None = None) -> float:
        """Record initial capital. Only writes to DB once (immutable)."""
        current_value = self.client.get_portfolio_value(prices)

        record = self.db.get_capital_record()
        if record is not None:
            self._initial_capital = record.initial_capital_usd
            self._capital_floor = record.capital_floor_usd
            logger.info(
                "Loaded existing capital record: initial=$%.2f  floor=$%.2f",
                self._initial_capital,
                self._capital_floor,
            )
        else:
            self._initial_capital = current_value
            self._capital_floor = current_value * (1 - self.config.max_loss_pct)
            self.db.set_capital_record(self._initial_capital, self.config.max_loss_pct)
            logger.info(
                "Recorded initial capital: $%.2f  floor=$%.2f (max loss %.0f%%)",
                self._initial_capital,
                self._capital_floor,
                self.config.max_loss_pct * 100,
            )

        self._peak_value = max(current_value, self.db.get_peak_value())
        self._daily_start_value = current_value
        self._daily_start_time = dt.datetime.now(dt.timezone.utc)
        self._hourly_start_value = current_value
        self._hourly_start_time = dt.datetime.now(dt.timezone.utc)

        return current_value

    # ------------------------------------------------------------------
    # Layer 3: Progressive drawdown throttling
    # ------------------------------------------------------------------

    def _update_state(self, current_value: float) -> TradingState:
        """Determine trading state based on drawdown from peak."""
        if self._state == TradingState.SHUTDOWN:
            return TradingState.SHUTDOWN

        if self._peak_value <= 0:
            return TradingState.NORMAL

        if current_value > self._peak_value:
            self._peak_value = current_value

        drawdown = 1 - (current_value / self._peak_value)
        drawdown_from_initial = 1 - (current_value / self._initial_capital)

        # Layer 2: Hard floor check
        if current_value <= self._capital_floor:
            self._state = TradingState.EMERGENCY
            logger.critical(
                "CAPITAL FLOOR BREACHED: $%.2f <= $%.2f — EMERGENCY SHUTDOWN",
                current_value,
                self._capital_floor,
            )
            return TradingState.EMERGENCY

        # Layer 4: Daily/hourly checks
        now = dt.datetime.now(dt.timezone.utc)

        if (now - self._daily_start_time).total_seconds() >= 86400:
            self._daily_start_value = current_value
            self._daily_start_time = now

        if (now - self._hourly_start_time).total_seconds() >= 3600:
            self._hourly_start_value = current_value
            self._hourly_start_time = now

        if self._daily_start_value > 0:
            daily_loss = 1 - (current_value / self._daily_start_value)
            if daily_loss >= self.config.daily_loss_limit_pct:
                if self._state != TradingState.HALTED_DAILY:
                    logger.warning(
                        "Daily loss limit hit: %.2f%% — halting until next UTC midnight",
                        daily_loss * 100,
                    )
                self._state = TradingState.HALTED_DAILY
                return TradingState.HALTED_DAILY

        if self._hourly_start_value > 0:
            hourly_loss = 1 - (current_value / self._hourly_start_value)
            if hourly_loss >= self.config.hourly_loss_limit_pct:
                if self._state != TradingState.HALTED_HOURLY:
                    logger.warning(
                        "Hourly loss limit hit: %.2f%% — pausing for 1 hour",
                        hourly_loss * 100,
                    )
                self._state = TradingState.HALTED_HOURLY
                return TradingState.HALTED_HOURLY

        # Layer 3: Progressive drawdown
        if drawdown >= 0.40:
            self._state = TradingState.EMERGENCY
        elif drawdown >= 0.30:
            self._state = TradingState.SELL_ONLY
        elif drawdown >= 0.20:
            self._state = TradingState.THROTTLED_25
        elif drawdown >= 0.10:
            self._state = TradingState.THROTTLED_50
        else:
            self._state = TradingState.NORMAL

        return self._state

    def get_position_scale(self) -> float:
        """Get the current position sizing multiplier based on state."""
        scales = {
            TradingState.NORMAL: 1.0,
            TradingState.THROTTLED_50: 0.5,
            TradingState.THROTTLED_25: 0.25,
            TradingState.SELL_ONLY: 0.1,
            TradingState.HALTED_DAILY: 0.0,
            TradingState.HALTED_HOURLY: 0.0,
            TradingState.EMERGENCY: 0.0,
            TradingState.SHUTDOWN: 0.0,
        }
        return scales.get(self._state, 0.0)

    def get_min_signal_strength(self) -> float:
        """Higher drawdown => require higher confidence signals."""
        thresholds = {
            TradingState.NORMAL: 0.6,
            TradingState.THROTTLED_50: 0.65,
            TradingState.THROTTLED_25: 0.75,
            TradingState.SELL_ONLY: 1.0,  # no buys allowed
        }
        return thresholds.get(self._state, 1.0)

    # ------------------------------------------------------------------
    # Layer 4: Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        portfolio_value: float,
        signal_strength: float,
        atr: float,
        price: float,
    ) -> float:
        """Calculate position size in USD using half-Kelly with ATR-based risk.

        Returns 0 if any limit would be breached.
        """
        scale = self.get_position_scale()
        if scale <= 0:
            return 0.0

        # Half-Kelly approximation: size proportional to edge/odds
        edge = max(signal_strength - 0.5, 0.0) * 2  # normalize to [0, 1]
        kelly_fraction = edge * 0.5  # half-Kelly

        base_size = portfolio_value * kelly_fraction
        max_size = portfolio_value * self.config.max_position_pct * scale

        size = min(base_size, max_size)

        # ATR-based risk check: don't risk more than position_pct on a stop-loss hit
        if atr > 0 and price > 0:
            risk_per_unit = atr * self.config.stop_loss_atr_mult
            max_risk = portfolio_value * self.config.max_position_pct * scale
            atr_size = (max_risk / risk_per_unit) * price
            size = min(size, atr_size)

        return max(size, 0.0)

    # ------------------------------------------------------------------
    # Layer 5: Pre-trade validation
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        product_id: str,
        side: str,
        size_usd: float,
        portfolio_value: float,
        current_positions: list[dict],
        coin_universe: list[str],
    ) -> RiskCheckResult:
        """Validate a proposed trade against all risk limits.

        Returns RiskCheckResult indicating whether the trade is allowed.
        """
        with self._lock:
            # Check trading state
            if self._state in (
                TradingState.EMERGENCY,
                TradingState.SHUTDOWN,
                TradingState.HALTED_DAILY,
                TradingState.HALTED_HOURLY,
            ):
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Trading halted: {self._state.value}",
                )

            # Sell-only mode check
            if self._state == TradingState.SELL_ONLY and side == "BUY":
                return RiskCheckResult(
                    allowed=False,
                    reason="Sell-only mode active (30-40% drawdown)",
                )

            # Check coin is in approved universe
            if product_id not in coin_universe:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"{product_id} not in approved trading universe",
                )

            if side == "SELL":
                return RiskCheckResult(allowed=True, max_size_usd=size_usd)

            # --- BUY-side checks ---

            # Max open positions
            open_count = len([p for p in current_positions if p.get("is_open")])
            if open_count >= self.config.max_open_positions:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Max open positions ({self.config.max_open_positions}) reached",
                )

            # Position size limit
            scale = self.get_position_scale()
            max_size = portfolio_value * self.config.max_position_pct * scale
            if size_usd > max_size:
                size_usd = max_size

            # Concentration limit
            existing_exposure = sum(
                p.get("value_usd", 0)
                for p in current_positions
                if p.get("product_id") == product_id and p.get("is_open")
            )
            max_concentration = portfolio_value * self.config.max_concentration_pct
            remaining_room = max_concentration - existing_exposure
            if remaining_room <= 0:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Concentration limit ({self.config.max_concentration_pct*100:.0f}%) reached for {product_id}",
                )
            size_usd = min(size_usd, remaining_room)

            # Cash balance check — include EURC that can be converted
            cash = self.client.get_usd_balance()
            if self._fx:
                try:
                    eurc_bal = self._fx.get_eurc_balance()
                    rate = self._fx.get_eur_usd_rate()
                    cash += eurc_bal * rate * (1 - self._fx.maker_fee)
                except Exception:
                    pass
            if size_usd > cash:
                size_usd = cash
            if size_usd <= 0:
                return RiskCheckResult(
                    allowed=False,
                    reason="Insufficient USD balance",
                )

            return RiskCheckResult(
                allowed=True,
                max_size_usd=size_usd,
                position_scale=scale,
            )

    # ------------------------------------------------------------------
    # Layer 6: Watchdog thread
    # ------------------------------------------------------------------

    def start_watchdog(self) -> None:
        """Start the independent portfolio monitoring thread."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="risk-watchdog"
        )
        self._watchdog_thread.start()
        logger.info(
            "Watchdog started (checking every %ds)", self.config.watchdog_interval_sec
        )

    def stop_watchdog(self) -> None:
        self._watchdog_running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)

    def _watchdog_loop(self) -> None:
        while self._watchdog_running:
            try:
                current_value = self.client.get_portfolio_value()

                # Guard against underreported values from transient API failures:
                # if value looks dangerously low, retry once before acting on it
                if current_value <= self._capital_floor and self._state != TradingState.EMERGENCY:
                    time.sleep(2)
                    retry_value = self.client.get_portfolio_value()
                    if retry_value > current_value:
                        logger.warning(
                            "Watchdog: portfolio value jumped $%.2f → $%.2f on retry (API hiccup?)",
                            current_value, retry_value,
                        )
                        current_value = retry_value

                fx_rate = None
                if self._fx:
                    try:
                        fx_rate = self._fx.get_eur_usd_rate()
                    except Exception:
                        pass

                snapshot = PortfolioSnapshot(
                    total_value_usd=current_value,
                    cash_usd=self.client.get_usd_balance(),
                    holdings_value_usd=current_value - self.client.get_usd_balance(),
                    num_positions=len(self.db.get_open_positions()),
                    drawdown_pct=(
                        1 - current_value / self._peak_value
                        if self._peak_value > 0
                        else 0
                    ),
                    eur_usd_rate=fx_rate,
                )
                self.db.save_snapshot(snapshot)

                old_state = self._state
                new_state = self._update_state(current_value)

                if new_state != old_state:
                    extra = ""
                    if self._fx:
                        try:
                            fx_s = self._fx.get_status(current_value)
                            extra = f"  EUR/USD={fx_s.eur_usd_rate:.4f}  €{fx_s.portfolio_value_eur:,.2f}  USD exp={fx_s.usd_exposure_pct*100:.0f}%"
                        except Exception:
                            pass
                    logger.warning(
                        "Watchdog: state changed %s -> %s  (value=$%.2f%s)",
                        old_state.value,
                        new_state.value,
                        current_value,
                        extra,
                    )

                if new_state == TradingState.EMERGENCY:
                    logger.critical("Watchdog: EMERGENCY — triggering shutdown")
                    if self._on_emergency:
                        self._on_emergency()
                    self._watchdog_running = False
                    return

            except Exception as e:
                logger.error("Watchdog error: %s", e)

            time.sleep(self.config.watchdog_interval_sec)

    # ------------------------------------------------------------------
    # Stop-loss / take-profit calculations
    # ------------------------------------------------------------------

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        return entry_price - (atr * self.config.stop_loss_atr_mult)

    def calculate_take_profit(self, entry_price: float, atr: float) -> float:
        return entry_price + (atr * self.config.take_profit_atr_mult)

    def update_trailing_stop(
        self, current_price: float, highest_price: float, atr: float
    ) -> float:
        """Recalculate trailing stop based on the highest price seen."""
        new_high = max(current_price, highest_price)
        return new_high - (atr * self.config.stop_loss_atr_mult)

    def check_stop_loss(self, current_price: float, stop_loss: float) -> bool:
        return current_price <= stop_loss

    def check_take_profit(self, current_price: float, take_profit: float) -> bool:
        return current_price >= take_profit
