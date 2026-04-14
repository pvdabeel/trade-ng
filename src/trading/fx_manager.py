"""EUR/USD currency risk management.

Parks idle capital in EURC (EUR stablecoin) and only converts to USDC
when needed for trades, limiting exposure to USD/EUR fluctuations.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass

from src.data.coinbase_client import CoinbaseClient

logger = logging.getLogger(__name__)

_RATE_CACHE_TTL = 60  # seconds


@dataclass
class FXStatus:
    eur_usd_rate: float
    portfolio_value_eur: float
    portfolio_value_usd: float
    usdc_balance: float
    eurc_balance: float
    usd_exposure_pct: float


class FXManager:
    """Manages EUR/USD currency risk by parking idle capital in EURC."""

    def __init__(
        self,
        client: CoinbaseClient,
        eurc_product_id: str = "EURC-USDC",
        max_usd_exposure_pct: float = 0.30,
        rebalance_threshold_pct: float = 0.05,
        min_rebalance_usd: float = 5.0,
        maker_fee_pct: float = 0.006,
        order_timeout_sec: int = 60,
    ):
        self.client = client
        self.eurc_product_id = eurc_product_id
        self.max_usd_exposure = max_usd_exposure_pct
        self.rebalance_threshold = rebalance_threshold_pct
        self.min_rebalance_usd = min_rebalance_usd
        self.maker_fee = maker_fee_pct
        self.timeout = order_timeout_sec

        self._rate_cache: float = 0.0
        self._rate_ts: float = 0.0

    # ------------------------------------------------------------------
    # Rate & exposure queries
    # ------------------------------------------------------------------

    def get_eur_usd_rate(self) -> float:
        """Live EUR/USD rate via EURC-USDC price. Cached for 60s."""
        now = time.time()
        if self._rate_cache > 0 and (now - self._rate_ts) < _RATE_CACHE_TTL:
            return self._rate_cache

        try:
            product = self.client.get_product(self.eurc_product_id)
            self._rate_cache = product.price
            self._rate_ts = now
            return self._rate_cache
        except Exception as e:
            logger.warning("Failed to fetch EUR/USD rate: %s", e)
            return self._rate_cache if self._rate_cache > 0 else 1.0

    def usd_to_eur(self, usd_amount: float) -> float:
        rate = self.get_eur_usd_rate()
        return usd_amount / rate if rate > 0 else usd_amount

    def get_eurc_balance(self) -> float:
        """Available EURC balance."""
        for acct in self.client.get_accounts():
            if acct.currency == "EURC":
                return acct.available
        return 0.0

    def get_status(self, portfolio_value_usd: float | None = None) -> FXStatus:
        """Full FX status snapshot.

        portfolio_value_usd should be the total portfolio value *including*
        EURC (as returned by CoinbaseClient.get_portfolio_value).
        """
        rate = self.get_eur_usd_rate()
        usdc = self.client.get_usd_balance()
        eurc = self.get_eurc_balance()
        eurc_in_usd = eurc * rate

        if portfolio_value_usd is None:
            portfolio_value_usd = self.client.get_portfolio_value()

        usd_portion = portfolio_value_usd - eurc_in_usd
        usd_exposure = (
            usd_portion / portfolio_value_usd if portfolio_value_usd > 0 else 1.0
        )

        return FXStatus(
            eur_usd_rate=rate,
            portfolio_value_eur=portfolio_value_usd / rate if rate > 0 else 0,
            portfolio_value_usd=portfolio_value_usd,
            usdc_balance=usdc,
            eurc_balance=eurc,
            usd_exposure_pct=max(0.0, usd_exposure),
        )

    # ------------------------------------------------------------------
    # Conversion operations
    # ------------------------------------------------------------------

    def conversion_cost(self, amount_usd: float) -> float:
        """Estimated fee for a single EURC<->USDC conversion."""
        return amount_usd * self.maker_fee

    def round_trip_cost(self, amount_usd: float) -> float:
        """Estimated total fee for USDC→EURC→USDC (park + unpark)."""
        return amount_usd * self.maker_fee * 2

    def ensure_usdc_for_trade(self, size_usd: float) -> bool:
        """Convert EURC -> USDC if needed to fund a trade.

        Accounts for the conversion fee so the trade receives the full
        amount requested.
        Returns True if sufficient USDC is available (or was made available).
        """
        usdc = self.client.get_usd_balance()
        if usdc >= size_usd:
            return True

        shortfall = size_usd - usdc
        if shortfall < self.min_rebalance_usd:
            return False

        eurc_available = self.get_eurc_balance()
        rate = self.get_eur_usd_rate()
        # We need enough EURC to cover shortfall + conversion fee + slippage
        gross_needed = shortfall / (1 - self.maker_fee)
        eurc_needed = gross_needed / rate if rate > 0 else gross_needed
        fee_est = shortfall * self.maker_fee / (1 - self.maker_fee)

        if eurc_available < eurc_needed:
            logger.warning(
                "Insufficient EURC (%.0f) to cover USDC shortfall "
                "(need %.0f EURC incl. ~$%.2f fee)",
                eurc_available, eurc_needed, fee_est,
            )
            return False

        logger.info(
            "FX: converting %.0f EURC → $%.2f USDC (fee ~$%.2f)",
            eurc_needed, shortfall, fee_est,
        )
        return self._sell_eurc_for_usdc(eurc_needed)

    def rebalance_to_eurc(self) -> bool:
        """Sweep excess USDC to EURC if USD exposure exceeds the cap.

        Only sweeps if the amount is large enough that conversion fees
        are justified relative to the FX protection gained.
        Returns True if a rebalance was performed (or none was needed).
        """
        status = self.get_status()

        target_exposure = self.max_usd_exposure
        if status.usd_exposure_pct <= target_exposure + self.rebalance_threshold:
            return True

        excess_pct = status.usd_exposure_pct - target_exposure
        excess_usd = excess_pct * status.portfolio_value_usd

        fee_cost = self.round_trip_cost(excess_usd)
        if excess_usd < self.min_rebalance_usd or excess_usd <= fee_cost * 10:
            return True

        usdc_available = status.usdc_balance
        # Deduct the conversion fee from what we sweep
        sweep_usd = min(excess_usd, usdc_available)
        usdc_after_fee = sweep_usd * (1 - self.maker_fee)
        if usdc_after_fee < self.min_rebalance_usd:
            return True

        rate = status.eur_usd_rate
        eurc_to_buy = int(usdc_after_fee / rate) if rate > 0 else 0
        if eurc_to_buy <= 0:
            return True

        fee_est = sweep_usd * self.maker_fee
        logger.info(
            "FX rebalance: $%.2f USDC → %d EURC (fee ~$%.2f, exposure %.0f%% → target %.0f%%)",
            sweep_usd, eurc_to_buy, fee_est,
            status.usd_exposure_pct * 100,
            target_exposure * 100,
        )
        return self._buy_eurc_with_usdc(eurc_to_buy)

    # ------------------------------------------------------------------
    # Order helpers (EURC-USDC pair)
    # ------------------------------------------------------------------

    def _buy_eurc_with_usdc(self, eurc_amount: int) -> bool:
        """Buy EURC using USDC (limit order on EURC-USDC)."""
        if eurc_amount <= 0:
            return True
        try:
            rate = self.get_eur_usd_rate()
            price_str = f"{rate * 1.002:.4f}"
            size_str = str(eurc_amount)
            cost_usd = eurc_amount * rate
            fee_est = cost_usd * self.maker_fee

            logger.info(
                "FX: buying %s EURC @ %s USDC (cost ~$%.2f, fee ~$%.2f)",
                size_str, price_str, cost_usd, fee_est,
            )
            result = self.client.place_limit_buy(
                product_id=self.eurc_product_id,
                size=size_str,
                price=price_str,
            )
            if result.status == "FAILED":
                logger.warning("FX buy EURC failed: %s", result.raw)
                return False

            filled, fill_data = self._wait_for_fill(result.order_id)
            if not filled:
                self.client.cancel_orders([result.order_id])
                logger.warning("FX buy EURC timed out, cancelled")
                return False

            actual_fee = self._extract_fee(fill_data, fee_est)
            logger.info("FX: bought %s EURC (fee=$%.4f)", size_str, actual_fee)
            return True
        except Exception as e:
            logger.error("FX buy EURC error: %s", e)
            return False

    def _sell_eurc_for_usdc(self, eurc_amount: float) -> bool:
        """Sell EURC for USDC (limit order on EURC-USDC)."""
        eurc_int = math.ceil(eurc_amount)
        if eurc_int <= 0:
            return False
        try:
            rate = self.get_eur_usd_rate()
            price_str = f"{rate * 0.998:.4f}"
            size_str = str(eurc_int)
            proceeds_usd = eurc_int * rate
            fee_est = proceeds_usd * self.maker_fee

            logger.info(
                "FX: selling %s EURC @ %s USDC (proceeds ~$%.2f, fee ~$%.2f)",
                size_str, price_str, proceeds_usd, fee_est,
            )
            result = self.client.place_limit_sell(
                product_id=self.eurc_product_id,
                size=size_str,
                price=price_str,
            )
            if result.status == "FAILED":
                logger.warning("FX sell EURC failed: %s", result.raw)
                return False

            filled, fill_data = self._wait_for_fill(result.order_id)
            if not filled:
                self.client.cancel_orders([result.order_id])
                logger.warning("FX sell EURC timed out, cancelled")
                return False

            actual_fee = self._extract_fee(fill_data, fee_est)
            logger.info("FX: sold %s EURC for USDC (fee=$%.4f)", size_str, actual_fee)
            return True
        except Exception as e:
            logger.error("FX sell EURC error: %s", e)
            return False

    def _wait_for_fill(self, order_id: str) -> tuple[bool, dict]:
        """Poll until the EURC conversion order fills or times out."""
        if not order_id:
            logger.info("FX: no order ID returned — treating as immediate fill")
            return True, {}

        not_found_count = 0
        time.sleep(0.5)

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
            except Exception:
                not_found_count += 1
                if not_found_count >= 3:
                    logger.info(
                        "FX order %s not found after %d attempts — likely filled instantly",
                        order_id, not_found_count,
                    )
                    return True, {}
            time.sleep(2)
        return False, {}

    @staticmethod
    def _extract_fee(fill_data: dict, estimated: float) -> float:
        try:
            fee_str = fill_data.get("total_fees", "")
            if fee_str:
                return float(fee_str)
        except (ValueError, TypeError):
            pass
        return estimated
