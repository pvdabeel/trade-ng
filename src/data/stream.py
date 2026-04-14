"""Real-time WebSocket price feed from Coinbase."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import websockets

logger = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"


@dataclass
class TickerUpdate:
    product_id: str
    price: float
    volume_24h: float
    timestamp: float


class PriceStream:
    """Async WebSocket stream for real-time Coinbase price updates."""

    def __init__(
        self,
        product_ids: list[str],
        api_key: str = "",
        api_secret: str = "",
    ):
        self.product_ids = product_ids
        self._api_key = api_key
        self._api_secret = api_secret
        self._prices: dict[str, float] = {}
        self._callbacks: list[Callable[[TickerUpdate], None]] = []
        self._running = False

    @property
    def prices(self) -> dict[str, float]:
        return dict(self._prices)

    def on_ticker(self, callback: Callable[[TickerUpdate], None]) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error("WebSocket error: %s — reconnecting in 5s", e)
                await asyncio.sleep(5)

    def stop(self) -> None:
        self._running = False

    async def _connect(self) -> None:
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channel": "ticker",
        }
        if self._api_key:
            subscribe_msg["api_key"] = self._api_key

        async with websockets.connect(COINBASE_WS_URL) as ws:
            await ws.send(json.dumps(subscribe_msg))
            logger.info("WebSocket connected, subscribed to %d products", len(self.product_ids))

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw_msg)
                    self._handle_message(msg)
                except json.JSONDecodeError:
                    continue

    def _handle_message(self, msg: dict) -> None:
        channel = msg.get("channel", "")
        if channel != "ticker":
            return

        for event in msg.get("events", []):
            for ticker in event.get("tickers", []):
                product_id = ticker.get("product_id", "")
                try:
                    price = float(ticker.get("price", 0))
                except (ValueError, TypeError):
                    continue

                if price <= 0:
                    continue

                volume = float(ticker.get("volume_24_h", 0))
                update = TickerUpdate(
                    product_id=product_id,
                    price=price,
                    volume_24h=volume,
                    timestamp=time.time(),
                )
                self._prices[product_id] = price

                for cb in self._callbacks:
                    try:
                        cb(update)
                    except Exception as e:
                        logger.error("Ticker callback error: %s", e)
