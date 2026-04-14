"""WebSocket endpoint for live dashboard updates."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.data.database import Database
from src.trading.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)

ws_router = APIRouter()

_connections: list[WebSocket] = []


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.remove(websocket)


async def broadcast(data: dict) -> None:
    """Send data to all connected WebSocket clients."""
    message = json.dumps(data)
    disconnected = []
    for ws in _connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _connections.remove(ws)
