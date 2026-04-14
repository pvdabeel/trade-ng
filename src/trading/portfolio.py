"""Portfolio state and P&L tracking."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from src.data.coinbase_client import CoinbaseClient
from src.data.database import Database, Position, Trade

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    product_id: str
    side: str
    entry_price: float
    size: float
    current_price: float
    stop_loss: float
    take_profit: float
    highest_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    value_usd: float
    is_open: bool
    strategy: str = "ml"
    fee: float = 0.0
    opened_at: str | None = None
    eval_hold_remaining: int | None = None

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "value_usd": self.value_usd,
            "is_open": self.is_open,
            "strategy": self.strategy,
            "fee": self.fee,
            "opened_at": self.opened_at,
            "eval_hold_remaining": self.eval_hold_remaining,
        }


@dataclass
class PortfolioSummary:
    total_value_usd: float
    cash_usd: float
    holdings_value_usd: float
    num_open_positions: int
    total_unrealized_pnl: float
    positions: list[PositionInfo]


class PortfolioTracker:
    """Tracks portfolio state by combining DB positions with live prices."""

    def __init__(self, client: CoinbaseClient, db: Database, eval_hold_sec: int = 900):
        self.client = client
        self.db = db
        self._eval_hold_sec = eval_hold_sec

    def get_summary(self, prices: dict[str, float] | None = None) -> PortfolioSummary:
        """Build a full portfolio summary with current prices."""
        positions_db = self.db.get_open_positions()
        cash = self.client.get_usd_balance()

        fee_by_product: dict[str, float] = {}
        try:
            from sqlalchemy import func
            with self.db.session() as s:
                for pos in positions_db:
                    q = s.query(func.sum(Trade.fee)).filter(
                        Trade.product_id == pos.product_id,
                        Trade.side == "BUY",
                    )
                    if pos.opened_at:
                        q = q.filter(Trade.created_at >= pos.opened_at)
                    result = q.scalar()
                    if result:
                        fee_by_product[pos.product_id] = result
        except Exception:
            pass

        position_infos = []
        total_holdings = 0.0
        total_unrealized = 0.0

        for pos in positions_db:
            current_price = self._get_price(pos.product_id, prices)
            if current_price <= 0:
                continue

            value = pos.size * current_price
            if value < 1.0:
                continue
            unrealized = (current_price - pos.entry_price) * pos.size
            unrealized_pct = (
                (current_price / pos.entry_price - 1) * 100
                if pos.entry_price > 0
                else 0
            )

            opened_iso = None
            eval_remaining = None
            if pos.opened_at:
                opened_iso = pos.opened_at.isoformat()
                age = (dt.datetime.utcnow() - pos.opened_at).total_seconds()
                remaining = self._eval_hold_sec - age
                if remaining > 0:
                    eval_remaining = int(remaining)

            info = PositionInfo(
                product_id=pos.product_id,
                side=pos.side,
                entry_price=pos.entry_price,
                size=pos.size,
                current_price=current_price,
                stop_loss=pos.stop_loss or 0,
                take_profit=pos.take_profit or 0,
                highest_price=pos.highest_price or pos.entry_price,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=unrealized_pct,
                value_usd=value,
                is_open=True,
                strategy=getattr(pos, "strategy", "ml") or "ml",
                fee=fee_by_product.get(pos.product_id, 0.0),
                opened_at=opened_iso,
                eval_hold_remaining=eval_remaining,
            )
            position_infos.append(info)
            total_holdings += value
            total_unrealized += unrealized

        total_value = self.client.get_portfolio_value(prices)

        return PortfolioSummary(
            total_value_usd=total_value,
            cash_usd=cash,
            holdings_value_usd=total_holdings,
            num_open_positions=len(position_infos),
            total_unrealized_pnl=total_unrealized,
            positions=position_infos,
        )

    def _get_price(self, product_id: str, prices: dict[str, float] | None) -> float:
        if prices and product_id in prices:
            return prices[product_id]
        try:
            return self.client.get_ticker(product_id)
        except Exception:
            return 0.0
