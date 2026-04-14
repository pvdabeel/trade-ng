from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    granularity = Column(String, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("product_id", "timestamp", "granularity", name="uq_candle"),
    )


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String, unique=True, nullable=False, index=True)
    product_id = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # BUY / SELL
    size = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    status = Column(String, nullable=False)  # PENDING / FILLED / CANCELLED
    signal_strength = Column(Float)
    strategy = Column(String, default="ml", index=True)  # "ml" or "momentum"
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    filled_at = Column(DateTime)
    notes = Column(Text)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, unique=True, nullable=False, index=True)
    side = Column(String, nullable=False, default="LONG")
    entry_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    highest_price = Column(Float)
    strategy = Column(String, default="ml", index=True)  # "ml" or "momentum"
    opened_at = Column(DateTime, default=dt.datetime.utcnow)
    closed_at = Column(DateTime)
    is_open = Column(Boolean, default=True, index=True)
    pnl = Column(Float)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow, index=True)
    total_value_usd = Column(Float, nullable=False)
    cash_usd = Column(Float, nullable=False)
    holdings_value_usd = Column(Float, nullable=False)
    num_positions = Column(Integer, nullable=False)
    drawdown_pct = Column(Float)
    eur_usd_rate = Column(Float)


class CapitalRecord(Base):
    """Immutable record of initial capital — written once, never modified."""

    __tablename__ = "capital_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=dt.datetime.utcnow)
    initial_capital_usd = Column(Float, nullable=False)
    max_loss_pct = Column(Float, nullable=False)
    capital_floor_usd = Column(Float, nullable=False)


class BlockedCoin(Base):
    __tablename__ = "blocked_coins"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, unique=True, nullable=False, index=True)
    blocked_at = Column(DateTime, default=dt.datetime.utcnow)
    reason = Column(String, default="manual")


class Database:
    def __init__(self, db_path: str = "data/trade.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema (safe to re-run)."""
        import sqlalchemy as sa

        with self.engine.connect() as conn:
            inspector = sa.inspect(self.engine)
            migrations = [
                ("trades", "strategy", "TEXT DEFAULT 'ml'"),
                ("positions", "strategy", "TEXT DEFAULT 'ml'"),
                ("portfolio_snapshots", "eur_usd_rate", "REAL"),
            ]
            for table, column, col_def in migrations:
                if table in inspector.get_table_names():
                    cols = [c["name"] for c in inspector.get_columns(table)]
                    if column not in cols:
                        conn.execute(
                            sa.text(
                                f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
                            )
                        )
                        conn.commit()

    def session(self) -> Session:
        return self._session_factory()

    # ------------------------------------------------------------------
    # Capital record helpers
    # ------------------------------------------------------------------

    def get_capital_record(self) -> CapitalRecord | None:
        with self.session() as s:
            return s.query(CapitalRecord).first()

    def set_capital_record(
        self, initial_capital: float, max_loss_pct: float
    ) -> CapitalRecord:
        with self.session() as s:
            existing = s.query(CapitalRecord).first()
            if existing is not None:
                return existing
            rec = CapitalRecord(
                initial_capital_usd=initial_capital,
                max_loss_pct=max_loss_pct,
                capital_floor_usd=initial_capital * (1 - max_loss_pct),
            )
            s.add(rec)
            s.commit()
            s.refresh(rec)
            return rec

    def reset_capital_record(self) -> None:
        """Delete existing capital record so it will be re-initialized on next start."""
        with self.session() as s:
            s.query(CapitalRecord).delete()
            s.commit()

    def reset_peak_value(self, new_peak: float | None = None) -> None:
        """Cap all snapshot values to new_peak so drawdown calculations reset.

        If new_peak is None, uses the most recent snapshot value.
        Preserves historical data for chart display.
        """
        with self.session() as s:
            if new_peak is None:
                latest = (
                    s.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.desc())
                    .first()
                )
                new_peak = latest.total_value_usd if latest else 0.0
            if new_peak > 0:
                s.query(PortfolioSnapshot).filter(
                    PortfolioSnapshot.total_value_usd > new_peak
                ).update({"total_value_usd": new_peak})
                s.commit()

    # ------------------------------------------------------------------
    # Portfolio snapshot helpers
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        with self.session() as s:
            s.add(snapshot)
            s.commit()

    def get_peak_value(self) -> float:
        with self.session() as s:
            from sqlalchemy import func

            result = s.query(func.max(PortfolioSnapshot.total_value_usd)).scalar()
            return result or 0.0

    def get_snapshots_since(
        self, since: dt.datetime
    ) -> list[PortfolioSnapshot]:
        with self.session() as s:
            return (
                s.query(PortfolioSnapshot)
                .filter(PortfolioSnapshot.timestamp >= since)
                .order_by(PortfolioSnapshot.timestamp)
                .all()
            )

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------

    def save_trade(self, trade: Trade) -> None:
        with self.session() as s:
            s.add(trade)
            s.commit()

    def update_trade(self, order_id: str, **kwargs) -> None:
        with self.session() as s:
            trade = s.query(Trade).filter_by(order_id=order_id).first()
            if trade:
                for k, v in kwargs.items():
                    setattr(trade, k, v)
                s.commit()

    def get_recent_trades(self, limit: int = 50) -> list[Trade]:
        with self.session() as s:
            return (
                s.query(Trade)
                .order_by(Trade.created_at.desc())
                .limit(limit)
                .all()
            )

    def get_recent_trades_for(
        self, product_id: str, since: dt.datetime, strategy: str | None = None,
    ) -> list[Trade]:
        """Return trades for a specific coin since a given time."""
        with self.session() as s:
            q = s.query(Trade).filter(
                Trade.product_id == product_id,
                Trade.created_at >= since,
            )
            if strategy:
                q = q.filter(Trade.strategy == strategy)
            return q.order_by(Trade.created_at.desc()).all()

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def get_open_positions(self, strategy: str | None = None) -> list[Position]:
        with self.session() as s:
            q = s.query(Position).filter_by(is_open=True)
            if strategy:
                q = q.filter_by(strategy=strategy)
            return q.all()

    def get_position(self, product_id: str) -> Position | None:
        with self.session() as s:
            return (
                s.query(Position)
                .filter_by(product_id=product_id, is_open=True)
                .first()
            )

    def save_position(self, position: Position) -> None:
        with self.session() as s:
            old = s.query(Position).filter_by(product_id=position.product_id).first()
            if old:
                s.delete(old)
                s.flush()
            s.add(position)
            s.commit()

    def get_closed_positions(self) -> list[Position]:
        with self.session() as s:
            return (
                s.query(Position)
                .filter_by(is_open=False)
                .filter(Position.pnl.isnot(None))
                .order_by(Position.closed_at)
                .all()
            )

    def close_position(self, product_id: str, pnl: float) -> None:
        with self.session() as s:
            pos = (
                s.query(Position)
                .filter_by(product_id=product_id, is_open=True)
                .first()
            )
            if pos:
                pos.is_open = False
                pos.closed_at = dt.datetime.utcnow()
                pos.pnl = pnl
                s.commit()

    def update_position(self, product_id: str, **kwargs) -> None:
        with self.session() as s:
            pos = (
                s.query(Position)
                .filter_by(product_id=product_id, is_open=True)
                .first()
            )
            if pos:
                for k, v in kwargs.items():
                    setattr(pos, k, v)
                s.commit()

    # ------------------------------------------------------------------
    # Blocklist helpers
    # ------------------------------------------------------------------

    def block_coin(self, product_id: str, reason: str = "manual") -> bool:
        """Add a coin to the blocklist. Returns True if newly added."""
        with self.session() as s:
            existing = s.query(BlockedCoin).filter_by(product_id=product_id).first()
            if existing:
                return False
            s.add(BlockedCoin(product_id=product_id, reason=reason))
            s.commit()
            return True

    def unblock_coin(self, product_id: str) -> bool:
        """Remove a coin from the blocklist. Returns True if it was blocked."""
        with self.session() as s:
            n = s.query(BlockedCoin).filter_by(product_id=product_id).delete()
            s.commit()
            return n > 0

    def get_blocked_coins(self) -> list[BlockedCoin]:
        with self.session() as s:
            return (
                s.query(BlockedCoin)
                .order_by(BlockedCoin.blocked_at.desc())
                .all()
            )

    def is_coin_blocked(self, product_id: str) -> bool:
        with self.session() as s:
            return s.query(BlockedCoin).filter_by(product_id=product_id).first() is not None
