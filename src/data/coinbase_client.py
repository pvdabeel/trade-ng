"""Coinbase Advanced Trade API wrapper — spot-only by design.

This client deliberately omits any margin, leverage, or futures endpoints.
Only spot buy/sell operations are implemented to structurally guarantee
that the user cannot lose more than their deposited capital.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coinbase.rest import RESTClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_credentials() -> tuple[str, str]:
    """Load API credentials from env vars or a JSON key file.

    Supports three scenarios:
    1. COINBASE_KEY_FILE pointing to a downloaded cdp_api_key.json
    2. Direct COINBASE_API_KEY + COINBASE_API_SECRET env vars
    3. Both (key file takes precedence)
    """
    key_file = os.getenv("COINBASE_KEY_FILE", "")
    if key_file and Path(key_file).is_file():
        with open(key_file) as f:
            data = json.load(f)
        api_key = data.get("name") or data.get("id", "")
        api_secret = data.get("privateKey", "")
        logger.info("Loaded credentials from key file: %s", key_file)
        return api_key, api_secret

    return (
        os.getenv("COINBASE_API_KEY", ""),
        os.getenv("COINBASE_API_SECRET", ""),
    )


@dataclass
class AccountBalance:
    currency: str
    available: float
    hold: float

    @property
    def total(self) -> float:
        return self.available + self.hold


@dataclass
class Product:
    product_id: str
    base_currency: str
    quote_currency: str
    status: str
    price: float
    volume_24h: float
    base_min_size: float
    base_max_size: float
    quote_increment: float
    base_increment: float = 0.00000001


@dataclass
class CandleData:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderResult:
    order_id: str
    product_id: str
    side: str
    size: float
    price: float | None
    status: str
    raw: dict


class CoinbaseClient:
    """Spot-only wrapper around Coinbase Advanced Trade API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        if api_key and api_secret:
            self._key, self._secret = api_key, api_secret
        else:
            self._key, self._secret = _load_credentials()
        self._client = RESTClient(api_key=self._key, api_secret=self._secret)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(resp: Any) -> dict:
        """Convert SDK typed response to a plain dict."""
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        if isinstance(resp, dict):
            return resp
        return {}

    def get_accounts(self) -> list[AccountBalance]:
        accounts: list[AccountBalance] = []
        cursor: str | None = None
        while True:
            kwargs: dict = {"limit": 250}
            if cursor:
                kwargs["cursor"] = cursor
            resp = self._to_dict(self._client.get_accounts(**kwargs))
            for acct in resp.get("accounts", []):
                avail = float(acct.get("available_balance", {}).get("value", 0) or 0)
                hold = float(acct.get("hold", {}).get("value", 0) or 0)
                if avail + hold > 0:
                    accounts.append(
                        AccountBalance(
                            currency=acct.get("currency", ""),
                            available=avail,
                            hold=hold,
                        )
                    )
            if not resp.get("has_next"):
                break
            cursor = resp.get("cursor")
            if not cursor:
                break
        return accounts

    def get_usd_balance(self) -> float:
        """Available cash balance in USD (includes USD-pegged stablecoins like USDC)."""
        total = 0.0
        for acct in self.get_accounts():
            if acct.currency == "USD" or acct.currency in self._USD_STABLECOINS:
                total += acct.available
        return total

    _USD_STABLECOINS = {"USDC", "USDT", "DAI", "PYUSD", "GUSD", "PAX", "USDS", "USD1"}
    _EUR_STABLECOINS = {"EURC"}
    _SKIP_CURRENCIES: set[str] = set()

    def get_portfolio_value(self, prices: dict[str, float] | None = None) -> float:
        """Total portfolio value in USD (cash + holdings at market price).

        Includes EURC holdings valued via EURC-USDC rate.
        """
        accounts = self.get_accounts()
        total = 0.0
        for acct in accounts:
            if acct.currency == "USD" or acct.currency in self._USD_STABLECOINS:
                total += acct.total
            elif acct.currency in self._EUR_STABLECOINS:
                total = self._price_via_api(acct, total)
            elif acct.currency in self._SKIP_CURRENCIES:
                continue
            elif prices:
                usd_key = f"{acct.currency}-USD"
                usdc_key = f"{acct.currency}-USDC"
                if usd_key in prices:
                    total += acct.total * prices[usd_key]
                elif usdc_key in prices:
                    total += acct.total * prices[usdc_key]
                else:
                    total = self._price_via_api(acct, total)
            else:
                total = self._price_via_api(acct, total)
        return total

    def _price_via_api(self, acct, total: float) -> float:
        """Try to price an account balance via API lookup."""
        suffixes = ("-USDC", "-USD") if acct.currency in self._EUR_STABLECOINS else ("-USD", "-USDC")
        for suffix in suffixes:
            try:
                product = self.get_product(f"{acct.currency}{suffix}")
                value = acct.total * product.price
                logger.debug("Priced %s: %.4f × $%.6f = $%.2f", acct.currency, acct.total, product.price, value)
                return total + value
            except Exception:
                continue
        if acct.total > 0.01:
            logger.warning("Cannot price %s (balance=%.4f), skipping — portfolio value may be understated", acct.currency, acct.total)
        return total

    # ------------------------------------------------------------------
    # Products / Market Data
    # ------------------------------------------------------------------

    def get_product(self, product_id: str) -> Product:
        resp = self._to_dict(self._client.get_product(product_id))
        return self._parse_product(resp)

    def list_products(self, product_type: str = "SPOT") -> list[Product]:
        resp = self._to_dict(self._client.get_products(product_type=product_type))
        products = []
        for p in resp.get("products", []):
            try:
                products.append(self._parse_product(p))
            except (KeyError, ValueError):
                continue
        return products

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_HOUR",
    ) -> list[CandleData]:
        resp = self._to_dict(
            self._client.get_candles(
                product_id=product_id,
                start=str(start),
                end=str(end),
                granularity=granularity,
            )
        )
        candles = []
        for c in resp.get("candles", []):
            candles.append(
                CandleData(
                    timestamp=int(c["start"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                )
            )
        return sorted(candles, key=lambda x: x.timestamp)

    def get_ticker(self, product_id: str) -> float:
        product = self.get_product(product_id)
        return product.price

    # ------------------------------------------------------------------
    # Spot Orders — NO margin/leverage/futures
    # ------------------------------------------------------------------

    def place_limit_buy(
        self, product_id: str, size: str, price: str
    ) -> OrderResult:
        order_id = str(uuid.uuid4())
        resp = self._to_dict(
            self._client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side="BUY",
                order_configuration={
                    "limit_limit_gtc": {
                        "base_size": size,
                        "limit_price": price,
                    }
                },
            )
        )
        return self._parse_order(resp, product_id, "BUY", size, price)

    def place_limit_sell(
        self, product_id: str, size: str, price: str
    ) -> OrderResult:
        order_id = str(uuid.uuid4())
        resp = self._to_dict(
            self._client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side="SELL",
                order_configuration={
                    "limit_limit_gtc": {
                        "base_size": size,
                        "limit_price": price,
                    }
                },
            )
        )
        return self._parse_order(resp, product_id, "SELL", size, price)

    def place_market_buy(self, product_id: str, quote_size: str) -> OrderResult:
        """Market buy specifying the USD amount to spend."""
        order_id = str(uuid.uuid4())
        resp = self._to_dict(
            self._client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side="BUY",
                order_configuration={"market_market_ioc": {"quote_size": quote_size}},
            )
        )
        return self._parse_order(resp, product_id, "BUY", None, None)

    def place_market_sell(self, product_id: str, size: str) -> OrderResult:
        """Market sell for stop-loss and momentum exits."""
        order_id = str(uuid.uuid4())
        resp = self._to_dict(
            self._client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side="SELL",
                order_configuration={"market_market_ioc": {"base_size": size}},
            )
        )
        return self._parse_order(resp, product_id, "SELL", size, None)

    def cancel_orders(self, order_ids: list[str]) -> dict:
        if not order_ids:
            return {}
        return self._to_dict(self._client.cancel_orders(order_ids=order_ids))

    def get_order(self, order_id: str) -> dict:
        return self._to_dict(self._client.get_order(order_id=order_id))

    def list_open_orders(self, product_id: str | None = None) -> list[dict]:
        kwargs: dict[str, Any] = {}
        if product_id:
            kwargs["product_id"] = product_id
        resp = self._to_dict(
            self._client.list_orders(order_status=["OPEN"], **kwargs)
        )
        return resp.get("orders", [])

    def list_filled_orders(
        self, limit: int = 100, start_date: str | None = None
    ) -> list[dict]:
        """Return filled orders from Coinbase, newest first."""
        kwargs: dict[str, Any] = {
            "order_status": ["FILLED"],
            "limit": limit,
            "sort_by": "LAST_FILL_TIME",
        }
        if start_date:
            kwargs["start_date"] = start_date

        all_orders: list[dict] = []
        cursor: str | None = None

        while True:
            if cursor:
                kwargs["cursor"] = cursor
            resp = self._to_dict(self._client.list_orders(**kwargs))
            orders = resp.get("orders", [])
            all_orders.extend(orders)
            cursor = resp.get("cursor")
            if not cursor or not orders or len(all_orders) >= limit:
                break

        return all_orders[:limit]

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_product(data: dict) -> Product:
        return Product(
            product_id=data.get("product_id", ""),
            base_currency=data.get("base_currency_id", ""),
            quote_currency=data.get("quote_currency_id", ""),
            status=data.get("status", ""),
            price=float(data.get("price") or 0),
            volume_24h=float(data.get("volume_24h") or 0),
            base_min_size=float(data.get("base_min_size") or 0),
            base_max_size=float(data.get("base_max_size") or 1e12),
            quote_increment=float(data.get("quote_increment") or 0.01),
            base_increment=float(data.get("base_increment") or 0.00000001),
        )

    @staticmethod
    def _parse_order(
        resp: dict, product_id: str, side: str, size: str | None, price: str | None
    ) -> OrderResult:
        success = resp.get("success", False)
        order_data = resp.get("success_response", {}) if success else {}
        return OrderResult(
            order_id=order_data.get("order_id", ""),
            product_id=product_id,
            side=side,
            size=float(size) if size else 0.0,
            price=float(price) if price else None,
            status="PENDING" if success else "FAILED",
            raw=resp,
        )
