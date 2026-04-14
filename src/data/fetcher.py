"""Historical OHLCV data fetcher and coin universe discovery."""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
import yaml
from sqlalchemy import text

from src.data.coinbase_client import CoinbaseClient, Product
from src.data.database import Candle, Database

logger = logging.getLogger(__name__)

GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}

MAX_CANDLES_PER_REQUEST = 300


def load_coin_config(path: str = "config/coins.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class CoinUniverse:
    """Discovers and manages the set of tradeable coins."""

    def __init__(self, client: CoinbaseClient, config_path: str = "config/coins.yaml"):
        self.client = client
        self.config = load_coin_config(config_path)
        self._products: list[Product] = []
        self.quote_currency: str = "USD"

    def detect_quote_currency(self) -> str:
        """Auto-detect whether to trade USD or USDC pairs based on account balance."""
        accounts = self.client.get_accounts()
        usd_bal = sum(a.available for a in accounts if a.currency == "USD")
        usdc_bal = sum(a.available for a in accounts if a.currency == "USDC")
        if usdc_bal > usd_bal:
            logger.info("Detected USDC balance ($%.2f) > USD ($%.2f), using USDC pairs", usdc_bal, usd_bal)
            return "USDC"
        return "USD"

    def discover(self, quote_currency: str = "auto") -> list[str]:
        """Return list of product_ids (e.g. 'BTC-USD' or 'BTC-USDC') to trade."""
        mode = self.config.get("mode", "auto")

        if quote_currency == "auto":
            self.quote_currency = self.detect_quote_currency()
        else:
            self.quote_currency = quote_currency

        if mode == "manual":
            coins = self.config.get("manual_coins", [])
            if self.quote_currency == "USDC":
                coins = [c.replace("-USD", "-USDC") for c in coins]
            logger.info("Manual mode: trading %d coins (%s)", len(coins), self.quote_currency)
            return coins

        all_products = self.client.list_products(product_type="SPOT")
        exclude_raw = set(self.config.get("exclude", []))
        exclude = set()
        for ex in exclude_raw:
            exclude.add(ex)
            base = ex.split("-")[0]
            exclude.add(f"{base}-USD")
            exclude.add(f"{base}-USDC")
        min_vol = self.config.get("min_volume_usd", 100_000)

        self._products = [
            p
            for p in all_products
            if p.quote_currency == self.quote_currency
            and p.status == "online"
            and p.product_id not in exclude
            and p.volume_24h * p.price >= min_vol
        ]

        product_ids = [p.product_id for p in self._products]
        logger.info(
            "Auto-discovered %d %s pairs above $%s 24h volume",
            len(product_ids),
            self.quote_currency,
            f"{min_vol:,.0f}",
        )
        return product_ids

    def get_product_info(self, product_id: str) -> Product | None:
        for p in self._products:
            if p.product_id == product_id:
                return p
        return None


class HistoricalFetcher:
    """Fetches historical candle data and stores it in the database."""

    def __init__(self, client: CoinbaseClient, db: Database):
        self.client = client
        self.db = db

    def fetch_candles(
        self,
        product_id: str,
        granularity: str = "ONE_HOUR",
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """Fetch historical candles and persist to DB.  Returns a DataFrame."""
        interval = GRANULARITY_SECONDS.get(granularity, 3600)
        now = int(time.time())
        start = now - lookback_days * 86400
        all_candles = []

        current = start
        while current < now:
            chunk_end = min(current + MAX_CANDLES_PER_REQUEST * interval, now)
            try:
                candles = self.client.get_candles(
                    product_id=product_id,
                    start=current,
                    end=chunk_end,
                    granularity=granularity,
                )
                all_candles.extend(candles)
            except Exception as e:
                logger.warning(
                    "Error fetching %s candles for %s: %s",
                    granularity,
                    product_id,
                    e,
                )
            current = chunk_end
            time.sleep(0.1)  # rate-limit courtesy

        self._persist_candles(product_id, granularity, all_candles)

        df = self._candles_to_df(all_candles)
        logger.info(
            "Fetched %d candles for %s (%s)", len(df), product_id, granularity
        )
        return df

    def load_candles(
        self,
        product_id: str,
        granularity: str = "ONE_HOUR",
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """Load candles from DB; fetch missing data if needed."""
        with self.db.session() as s:
            cutoff = dt.datetime.utcnow() - dt.timedelta(days=lookback_days)
            rows = (
                s.query(Candle)
                .filter(
                    Candle.product_id == product_id,
                    Candle.granularity == granularity,
                    Candle.timestamp >= cutoff,
                )
                .order_by(Candle.timestamp)
                .all()
            )

        if not rows:
            return self.fetch_candles(product_id, granularity, lookback_days)

        df = pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )
        df.set_index("timestamp", inplace=True)
        return df

    def _persist_candles(self, product_id, granularity, candles) -> None:
        with self.db.session() as s:
            for c in candles:
                ts = dt.datetime.utcfromtimestamp(c.timestamp)
                existing = (
                    s.query(Candle)
                    .filter_by(
                        product_id=product_id,
                        timestamp=ts,
                        granularity=granularity,
                    )
                    .first()
                )
                if existing:
                    existing.open = c.open
                    existing.high = c.high
                    existing.low = c.low
                    existing.close = c.close
                    existing.volume = c.volume
                else:
                    s.add(
                        Candle(
                            product_id=product_id,
                            timestamp=ts,
                            granularity=granularity,
                            open=c.open,
                            high=c.high,
                            low=c.low,
                            close=c.close,
                            volume=c.volume,
                        )
                    )
            s.commit()

    @staticmethod
    def _candles_to_df(candles) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).set_index("timestamp")
        df = pd.DataFrame(
            [
                {
                    "timestamp": dt.datetime.utcfromtimestamp(c.timestamp),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        )
        df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
