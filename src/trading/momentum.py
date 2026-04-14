"""On-demand momentum trading strategy.

Scans for coins with >5% hourly price moves and trades them aggressively
using idle capital.  Activated/deactivated by the user via CLI or dashboard.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.coinbase_client import CoinbaseClient, OrderResult
from src.data.database import Database, Position, Trade
from src.trading.fx_manager import FXManager
from src.trading.risk import CooldownTracker, RiskManager, TradingState

logger = logging.getLogger(__name__)

FLAG_FILE = Path("data/momentum.flag")
SCAN_TRIGGER_FILE = Path("data/momentum_scan_trigger.flag")
SCAN_STATS_FILE = Path("data/momentum_scan_stats.json")
WATCHLIST_FILE = Path("data/momentum_watchlist.json")
DECISION_LOG_FILE = Path("data/decision_log.json")
_DECISION_LOG_MAX = 500

STRATEGY = "momentum"


@dataclass
class MomentumConfig:
    scan_interval_sec: int = 30
    min_hourly_change_pct: float = 0.05
    max_position_pct: float = 0.05
    max_open_positions: int = 5
    trailing_stop_pct: float = 0.02
    take_profit_pct: float = 0.05
    hard_stop_pct: float = 0.03
    max_hold_minutes: int = 120
    min_hold_seconds: int = 120
    evaluation_hold_sec: int = 900
    catastrophic_stop_pct: float = 0.10
    min_volume_usd: float = 500_000
    # Scale-in: gradual entry for very large moves
    scale_in_min_change_pct: float = 0.10
    scale_in_initial_pct: float = 0.25
    scale_in_step_pct: float = 0.25
    scale_in_max_pct: float = 1.50
    # Pullback re-entry: wait for correction before entering strong movers
    pullback_watch_min_pct: float = 0.06
    pullback_min_drop_pct: float = 0.03
    pullback_min_bounce_pct: float = 0.015
    pullback_watch_ttl_sec: int = 14400


@dataclass
class ScanStats:
    """Snapshot of the latest scan results (exposed via API)."""
    is_active: bool = False
    last_scan_time: str | None = None
    coins_scanned: int = 0
    candidates_found: int = 0
    open_positions: int = 0
    total_trades: int = 0


@dataclass
class _PriceEntry:
    price: float
    ts: float


@dataclass
class _EntryAnalysis:
    """Intraday price-action analysis for smart entry timing."""
    product_id: str
    current_price: float
    # Range: where the price sits within today's high-low (0.0 = at low, 1.0 = at high)
    range_position: float
    # How far price has pulled back from the intraday high (0.0 = at high)
    pullback_pct: float
    # Short-term trend: positive = rising, negative = falling (slope of last N candles)
    short_trend: float
    # RSI (14-period) — >70 overbought, <30 oversold
    rsi: float
    # Intraday volatility (std-dev of returns as fraction of price)
    volatility: float
    # Today's high / low for dynamic SL/TP
    intraday_high: float
    intraday_low: float
    # Recent volume relative to average (>1 = above average)
    volume_ratio: float
    # Final verdict
    signal: str  # "enter", "wait", "skip", "pullback_watch"
    reason: str


@dataclass
class _DecisionRecord:
    """Recorded entry decision with subsequent outcome tracking."""
    product_id: str
    timestamp: float
    decision: str          # "enter" | "skip" | "wait"
    price: float
    hourly_change_pct: float
    rsi: float
    range_position: float
    short_trend: float
    volatility: float
    reason: str
    price_after_15m: float | None = None
    price_after_1h: float | None = None
    price_after_2h: float | None = None
    outcome: str | None = None   # "correct" | "incorrect" | None
    realized_pnl_pct: float | None = None
    exit_reason: str | None = None
    hold_seconds: int | None = None
    entry_fee_pct: float | None = None


@dataclass
class _WatchlistEntry:
    """Candidate waiting for a better entry."""
    product_id: str
    hourly_change_pct: float
    added_at: float  # timestamp
    expiry: float  # timestamp — give up waiting after this


_WATCHLIST_TTL_SEC = 1800  # 30 min max wait for a better entry


@dataclass
class _PullbackWatch:
    """Tracks a strong mover through its pullback/recovery cycle."""
    product_id: str
    hourly_change_pct: float
    peak_price: float
    local_low: float
    phase: str              # "pullback" (waiting for drop) or "recovery" (drop happened, waiting for bounce)
    added_at: float
    expiry: float


@dataclass
class _ScaleInState:
    """Tracks gradual scale-in progress for a position."""
    product_id: str
    base_budget: float
    total_invested_usd: float
    tranches_done: int
    last_tranche_time: float


_UNTRADEABLE_ERRORS = frozenset([
    "account is not available",
    "target is not enabled for trading",
])

# Errors that warrant a temporary cooldown (minutes) rather than permanent block
_COOLDOWN_MINUTES = 30


class MomentumScanner:
    """Background scanner that detects and trades hourly momentum spikes.

    Two independent threads:
      - scan thread:    finds new momentum candidates and enters positions
      - monitor thread: checks exits (trailing stop, TP, time cap) on open
                        momentum positions; runs as long as positions exist
    """

    def __init__(
        self,
        client: CoinbaseClient,
        db: Database,
        risk: RiskManager,
        coin_ids: list[str],
        config: MomentumConfig | None = None,
        maker_fee_pct: float = 0.006,
        taker_fee_pct: float = 0.012,
        order_timeout_sec: int = 60,
        fx_manager: FXManager | None = None,
        cooldown: CooldownTracker | None = None,
    ):
        self.client = client
        self.db = db
        self.risk = risk
        self.coin_ids = coin_ids
        self.cfg = config or MomentumConfig()
        self.maker_fee = maker_fee_pct
        self.taker_fee = taker_fee_pct
        self.timeout = order_timeout_sec
        self.fx = fx_manager
        self._shared_cooldown = cooldown

        # Rolling price buffer: product_id -> deque of _PriceEntry
        self._prices: dict[str, deque[_PriceEntry]] = {}
        self._max_history = max(3600 // max(self.cfg.scan_interval_sec, 1) + 5, 130)

        self._scan_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._scan_stop = threading.Event()
        self._monitor_stop = threading.Event()
        self._lock = threading.Lock()
        self._stats = ScanStats()
        self._product_cache: dict = {}

        # Coins that can't be traded at all (permanent for this session)
        self._blacklist: set[str] = set()
        # Coins on temporary cooldown: product_id -> expiry timestamp
        self._cooldowns: dict[str, float] = {}
        # Candidates waiting for a better entry point
        self._watchlist: dict[str, _WatchlistEntry] = {}
        # Scale-in tracking for positions entered with gradual sizing
        self._scale_in: dict[str, _ScaleInState] = {}
        # Pullback re-entry: strong movers waiting for correction + recovery
        self._pullback_watches: dict[str, _PullbackWatch] = {}

        # Decision confidence tracking
        self._decision_log: list[_DecisionRecord] = self._load_decision_log()
        self._last_wait_log: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the scan thread (new entries) is active."""
        return self._scan_thread is not None and self._scan_thread.is_alive()

    @property
    def is_monitoring(self) -> bool:
        """True if the monitor thread (exit checks) is active."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def get_stats(self) -> ScanStats:
        with self._lock:
            s = ScanStats(
                is_active=self.is_running,
                last_scan_time=self._stats.last_scan_time,
                coins_scanned=self._stats.coins_scanned,
                candidates_found=self._stats.candidates_found,
                open_positions=len(self.db.get_open_positions(strategy=STRATEGY)),
                total_trades=self._stats.total_trades,
            )
        return s

    def get_watchlist(self) -> list[dict]:
        """Return the current watchlist for the dashboard with entry analysis."""
        now = time.time()
        result = []
        for w in sorted(self._watchlist.values(), key=lambda x: x.hourly_change_pct, reverse=True):
            price = None
            buf = self._prices.get(w.product_id)
            if buf:
                price = buf[-1].price
            ttl = max(0, int(w.expiry - now))
            entry: dict = {
                "product_id": w.product_id,
                "hourly_change_pct": w.hourly_change_pct,
                "current_price": price,
                "watching_sec": int(now - w.added_at),
                "ttl_sec": ttl,
                "rsi": None,
                "range_position": None,
                "short_trend": None,
                "signal": None,
                "reason": None,
                "pullback_pct": None,
                "intraday_high": None,
                "intraday_low": None,
                "scale_in": w.hourly_change_pct >= self.cfg.scale_in_min_change_pct,
            }
            if price and price > 0:
                try:
                    analysis = self._analyze_entry(w.product_id, price, w.hourly_change_pct)
                    entry["rsi"] = round(analysis.rsi, 1)
                    entry["range_position"] = round(analysis.range_position, 3)
                    entry["short_trend"] = round(analysis.short_trend, 4)
                    entry["signal"] = analysis.signal
                    entry["reason"] = analysis.reason
                    entry["pullback_pct"] = round(analysis.pullback_pct, 4)
                    entry["intraday_high"] = analysis.intraday_high
                    entry["intraday_low"] = analysis.intraday_low
                except Exception:
                    pass
            result.append(entry)

        # Append pullback-watch entries so the dashboard can show them
        for pw in sorted(self._pullback_watches.values(),
                         key=lambda x: x.hourly_change_pct, reverse=True):
            price = None
            buf = self._prices.get(pw.product_id)
            if buf:
                price = buf[-1].price
            ttl = max(0, int(pw.expiry - now))
            drop_pct = ((pw.peak_price - (price or pw.peak_price)) / pw.peak_price
                        if pw.peak_price > 0 else 0)
            bounce_pct = (((price or pw.local_low) - pw.local_low) / pw.local_low
                          if pw.local_low > 0 else 0)
            entry = {
                "product_id": pw.product_id,
                "hourly_change_pct": pw.hourly_change_pct,
                "current_price": price,
                "watching_sec": int(now - pw.added_at),
                "ttl_sec": ttl,
                "rsi": None,
                "range_position": None,
                "short_trend": None,
                "signal": "pullback_watch",
                "reason": f"phase={pw.phase}, drop={drop_pct:.1%}, bounce={bounce_pct:.1%}",
                "pullback_pct": round(drop_pct, 4),
                "intraday_high": pw.peak_price,
                "intraday_low": pw.local_low,
                "scale_in": pw.hourly_change_pct >= self.cfg.scale_in_min_change_pct,
                "pullback_phase": pw.phase,
                "pullback_drop_pct": round(drop_pct, 4),
                "pullback_bounce_pct": round(bounce_pct, 4),
                "pullback_peak": pw.peak_price,
                "pullback_low": pw.local_low,
            }
            if price and price > 0:
                try:
                    analysis = self._analyze_entry(pw.product_id, price, pw.hourly_change_pct)
                    entry["rsi"] = round(analysis.rsi, 1)
                    entry["range_position"] = round(analysis.range_position, 3)
                    entry["short_trend"] = round(analysis.short_trend, 4)
                except Exception:
                    pass
            result.append(entry)

        return result

    def _persist_watchlist(self) -> None:
        """Write watchlist to a JSON file so the dashboard process can read it."""
        try:
            data = self.get_watchlist()
            WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            WATCHLIST_FILE.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def load_watchlist_file() -> list[dict]:
        """Read the persisted watchlist (for the dashboard process)."""
        try:
            if WATCHLIST_FILE.exists():
                return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    def _persist_scan_stats(self) -> None:
        """Write scan stats to a JSON file so the dashboard process can read them."""
        try:
            with self._lock:
                data = {
                    "last_scan_time": self._stats.last_scan_time,
                    "coins_scanned": self._stats.coins_scanned,
                    "candidates_found": self._stats.candidates_found,
                }
            SCAN_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SCAN_STATS_FILE.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def load_scan_stats_file() -> dict:
        """Read persisted scan stats (for the dashboard process)."""
        try:
            if SCAN_STATS_FILE.exists():
                return json.loads(SCAN_STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Decision confidence tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _load_decision_log() -> list[_DecisionRecord]:
        try:
            if DECISION_LOG_FILE.exists():
                raw = json.loads(DECISION_LOG_FILE.read_text(encoding="utf-8"))
                return [_DecisionRecord(**r) for r in raw[-_DECISION_LOG_MAX:]]
        except Exception:
            pass
        return []

    def _save_decision_log(self) -> None:
        try:
            trimmed = self._decision_log[-_DECISION_LOG_MAX:]
            data = [asdict(r) for r in trimmed]
            DECISION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            DECISION_LOG_FILE.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def _log_decision(
        self, product_id: str, decision: str, price: float,
        hourly_change_pct: float, analysis: _EntryAnalysis,
    ) -> None:
        """Record a skip/wait/enter decision for later outcome evaluation."""
        now = time.time()
        if decision in ("wait", "pullback_watch"):
            last = self._last_wait_log.get(product_id, 0)
            if now - last < 600:
                return
            self._last_wait_log[product_id] = now

        rec = _DecisionRecord(
            product_id=product_id,
            timestamp=now,
            decision=decision,
            price=price,
            hourly_change_pct=hourly_change_pct,
            rsi=analysis.rsi,
            range_position=analysis.range_position,
            short_trend=analysis.short_trend,
            volatility=analysis.volatility,
            reason=analysis.reason,
        )
        with self._lock:
            self._decision_log.append(rec)
            if len(self._decision_log) > _DECISION_LOG_MAX:
                self._decision_log = self._decision_log[-_DECISION_LOG_MAX:]
        self._save_decision_log()

    def _measure_decision_outcomes(self) -> None:
        """Fetch prices for past decisions to evaluate correctness."""
        now = time.time()
        changed = False
        api_calls = 0

        for rec in self._decision_log:
            age = now - rec.timestamp
            if age < 900:
                continue
            # Fully measured — nothing left to fill
            if rec.price_after_2h is not None:
                continue

            price = None
            buf = self._prices.get(rec.product_id)
            if buf:
                price = buf[-1].price
            elif api_calls < 5:
                try:
                    price = self.client.get_ticker(rec.product_id)
                    api_calls += 1
                except Exception:
                    pass

            if price is None:
                if age > 10800:
                    rec.outcome = rec.outcome or "unknown"
                    changed = True
                continue

            if rec.price_after_15m is None and age >= 900:
                rec.price_after_15m = price
                changed = True
            if rec.price_after_1h is None and age >= 3600:
                rec.price_after_1h = price
                changed = True
            if rec.price_after_2h is None and age >= 7200:
                rec.price_after_2h = price
                changed = True

            # Evaluate / refine outcome whenever new price data arrives
            new_outcome = self._evaluate_decision(rec)
            if new_outcome != rec.outcome:
                rec.outcome = new_outcome
                changed = True

        if changed:
            self._save_decision_log()

    @staticmethod
    def _evaluate_decision(rec: _DecisionRecord) -> str:
        """Grade a past decision based on subsequent price movement.

        For a momentum strategy, the key question is: did we make money
        if we entered, and did we avoid a loss if we didn't?
        """
        ref = rec.price_after_2h or rec.price_after_1h or rec.price_after_15m
        if ref is None or rec.price <= 0:
            return "unknown"
        change = (ref - rec.price) / rec.price

        if rec.decision == "enter":
            # Correct if still above entry after trailing-stop distance (~2%)
            return "correct" if change > -0.02 else "incorrect"

        if rec.decision == "skip":
            # Correct if it didn't go up much (we didn't miss out)
            return "correct" if change < 0.03 else "incorrect"

        if rec.decision == "wait":
            # Correct if the price dipped first (we could enter lower)
            # OR if it didn't run away from us
            if rec.price_after_15m and rec.price > 0:
                dip = (rec.price_after_15m - rec.price) / rec.price
                if dip < -0.01:
                    return "correct"
            # Incorrect only if it ran significantly without us
            return "correct" if change < 0.03 else "incorrect"

        if rec.decision == "pullback_watch":
            # Correct if there was any dip (pullback materialized)
            if rec.price_after_15m and rec.price > 0:
                dip = (rec.price_after_15m - rec.price) / rec.price
                if dip < -0.01:
                    return "correct"
            # Also correct if it didn't keep running
            return "correct" if change < 0.03 else "incorrect"

        return "unknown"

    def get_confidence_scores(self) -> dict:
        """Compute accuracy per coin from historical outcomes."""
        return self._build_confidence(self._decision_log)

    @staticmethod
    def _build_confidence(records: list[_DecisionRecord]) -> dict:
        measured = [r for r in records if r.outcome in ("correct", "incorrect")]

        # Per-coin accuracy
        by_coin: dict[str, list[_DecisionRecord]] = {}
        for r in measured:
            by_coin.setdefault(r.product_id, []).append(r)
        coins: dict = {}
        for pid, recs in sorted(by_coin.items(), key=lambda x: len(x[1]), reverse=True):
            c = sum(1 for r in recs if r.outcome == "correct")
            coins[pid] = {"accuracy": round(c / len(recs), 3),
                          "correct": c, "total": len(recs)}

        # Overall
        if measured:
            c = sum(1 for r in measured if r.outcome == "correct")
            overall = {"accuracy": round(c / len(measured), 3),
                       "correct": c, "total": len(measured)}
        else:
            overall = {"accuracy": None, "correct": 0, "total": 0}

        recent = sorted(records, key=lambda r: r.timestamp, reverse=True)[:20]

        # Rolling accuracy time series (learning curve)
        window = 10
        chrono = sorted(measured, key=lambda r: r.timestamp)
        series: list[dict] = []
        for i, r in enumerate(chrono):
            win = chrono[max(0, i - window + 1): i + 1]
            correct_n = sum(1 for w in win if w.outcome == "correct")
            series.append({
                "timestamp": r.timestamp,
                "accuracy": round(correct_n / len(win), 3),
                "decision": r.decision,
                "outcome": r.outcome,
            })

        # Yield stats for "enter" decisions with realized outcomes
        entered = [r for r in records
                   if r.decision == "enter" and r.realized_pnl_pct is not None]
        if entered:
            yields = [r.realized_pnl_pct for r in entered]
            wins = [y for y in yields if y > 0]
            losses = [y for y in yields if y <= 0]
            hold_times = [r.hold_seconds for r in entered if r.hold_seconds is not None]
            by_exit: dict[str, list[float]] = {}
            for r in entered:
                key = r.exit_reason or "unknown"
                by_exit.setdefault(key, []).append(r.realized_pnl_pct)
            yield_stats: dict = {
                "avg_yield_pct": round(sum(yields) / len(yields), 4),
                "median_yield_pct": round(sorted(yields)[len(yields) // 2], 4),
                "win_rate": round(len(wins) / len(yields), 3),
                "total_trades": len(yields),
                "avg_hold_sec": round(sum(hold_times) / len(hold_times)) if hold_times else 0,
                "best_pct": round(max(yields), 4),
                "worst_pct": round(min(yields), 4),
                "by_exit": {
                    k: {"avg_pct": round(sum(v) / len(v), 4), "count": len(v)}
                    for k, v in by_exit.items()
                },
            }
            yield_series = [
                {"timestamp": r.timestamp, "yield_pct": r.realized_pnl_pct}
                for r in sorted(entered, key=lambda r: r.timestamp)
            ]
        else:
            yield_stats = {
                "avg_yield_pct": None, "median_yield_pct": None,
                "win_rate": None, "total_trades": 0,
                "avg_hold_sec": 0, "best_pct": None, "worst_pct": None,
                "by_exit": {},
            }
            yield_series = []

        return {
            "coins": coins,
            "overall": overall,
            "series": series,
            "yield_stats": yield_stats,
            "yield_series": yield_series,
            "recent": [
                {
                    "product_id": r.product_id,
                    "timestamp": r.timestamp,
                    "decision": r.decision,
                    "price": r.price,
                    "hourly_change_pct": r.hourly_change_pct,
                    "outcome": r.outcome,
                    "price_change_pct": round(
                        ((r.price_after_2h or r.price_after_1h or r.price_after_15m or r.price)
                         - r.price) / r.price, 4
                    ) if r.price > 0 else None,
                    "realized_pnl_pct": r.realized_pnl_pct,
                    "exit_reason": r.exit_reason,
                    "hold_seconds": r.hold_seconds,
                }
                for r in recent
            ],
        }

    @staticmethod
    def load_confidence_file() -> dict:
        """Read the persisted confidence scores (for the dashboard process)."""
        try:
            if not DECISION_LOG_FILE.exists():
                return {}
            raw = json.loads(DECISION_LOG_FILE.read_text(encoding="utf-8"))
            records = [_DecisionRecord(**r) for r in raw[-_DECISION_LOG_MAX:]]
            return MomentumScanner._build_confidence(records)
        except Exception:
            return {}

    def start(self) -> None:
        """Start scanning for new momentum trades (also starts monitor)."""
        if self.is_running:
            return
        self._seed_price_buffer()
        self._scan_stop.clear()
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True, name="momentum-scan")
        self._scan_thread.start()
        self._ensure_monitor()
        FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        FLAG_FILE.touch()
        logger.info("Momentum scanner started (interval=%ds)", self.cfg.scan_interval_sec)

    def stop(self) -> None:
        """Stop scanning for NEW trades. Monitor keeps running for existing positions
        and any pending scale-in adds."""
        if not self.is_running:
            return
        self._scan_stop.set()
        if self._scan_thread:
            self._scan_thread.join(timeout=10)
            self._scan_thread = None
        self._watchlist.clear()
        self._persist_watchlist()
        FLAG_FILE.unlink(missing_ok=True)
        pending_scale = len(self._scale_in)
        pending_pullback = len(self._pullback_watches)
        pending = pending_scale + pending_pullback
        if pending:
            logger.info(
                "Momentum scanner stopped (%d scale-in, %d pullback-watch continue via monitor)",
                pending_scale, pending_pullback,
            )
        else:
            logger.info("Momentum scanner stopped (monitor continues for open positions)")

    def stop_all(self) -> None:
        """Stop everything: scanner + monitor + scale-in + pullback tracking."""
        self.stop()
        with self._lock:
            self._scale_in.clear()
            self._pullback_watches.clear()
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None
        logger.info("Momentum monitor stopped")

    def _ensure_monitor(self) -> None:
        """Start the monitor thread if not already running."""
        if self.is_monitoring:
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="momentum-monitor")
        self._monitor_thread.start()

    def close_position(self, product_id: str) -> bool:
        """Manually close a single momentum position at market."""
        positions = self.db.get_open_positions(strategy=STRATEGY)
        pos = next((p for p in positions if p.product_id == product_id), None)
        if not pos:
            logger.warning("No open momentum position for %s", product_id)
            return False
        try:
            current_price = self.client.get_ticker(product_id)
        except Exception as e:
            logger.error("Cannot get price for %s: %s", product_id, e)
            return False
        self._exit_momentum(pos, current_price, "manual close")
        return True

    def close_all_positions(self) -> int:
        """Manually close ALL open momentum positions at market."""
        positions = self.db.get_open_positions(strategy=STRATEGY)
        if not positions:
            return 0
        closed = 0
        for pos in positions:
            try:
                current_price = self.client.get_ticker(pos.product_id)
                self._exit_momentum(pos, current_price, "manual close-all")
                closed += 1
            except Exception as e:
                logger.error("Failed to close momentum %s: %s", pos.product_id, e)
        return closed

    # ------------------------------------------------------------------
    # Flag file helpers (for cross-process activation)
    # ------------------------------------------------------------------

    @staticmethod
    def is_flag_active() -> bool:
        return FLAG_FILE.exists()

    @staticmethod
    def set_flag(active: bool) -> None:
        FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if active:
            FLAG_FILE.touch()
        else:
            FLAG_FILE.unlink(missing_ok=True)

    @staticmethod
    def request_scan() -> None:
        SCAN_TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCAN_TRIGGER_FILE.touch()

    @staticmethod
    def consume_scan_trigger() -> bool:
        if SCAN_TRIGGER_FILE.exists():
            SCAN_TRIGGER_FILE.unlink(missing_ok=True)
            return True
        return False

    def _is_blocked(self, product_id: str) -> bool:
        """True if the coin is blacklisted, on cooldown, or on the persistent blocklist."""
        if product_id in self._blacklist:
            return True
        if self._shared_cooldown and self._shared_cooldown.is_blocked(product_id):
            return True
        if self.db and self.db.is_coin_blocked(product_id):
            return True
        expiry = self._cooldowns.get(product_id)
        if expiry is not None:
            if time.time() < expiry:
                return True
            del self._cooldowns[product_id]
        return False

    def _add_cooldown(self, product_id: str, reason: str, minutes: int | None = None) -> None:
        """Put a coin on temporary cooldown after a loss or failed trade."""
        cd_min = minutes if minutes is not None else _COOLDOWN_MINUTES
        self._cooldowns[product_id] = time.time() + cd_min * 60
        if self._shared_cooldown:
            self._shared_cooldown.add(product_id, f"momentum {reason}", cd_min)
        else:
            logger.info("Momentum: %s on %dmin cooldown (%s)", product_id, cd_min, reason)

    def _check_recent_losses(self, product_id: str) -> tuple[bool, str]:
        """Check recent momentum trades for this coin.

        Returns (should_skip, reason).  Applies progressive cooldowns:
          - 1 recent loss  → 30 min cooldown
          - 2 recent losses → 60 min cooldown
          - 3+ recent losses → 120 min cooldown (skip for this session cycle)
        """
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)
        trades = self.db.get_recent_trades_for(product_id, since, strategy=STRATEGY)

        if not trades:
            return False, ""

        sells = [t for t in trades if t.side == "SELL" and t.status == "FILLED"]
        buys = [t for t in trades if t.side == "BUY" and t.status == "FILLED"]

        if not sells:
            return False, ""

        # Count losing sells (those with "stop-loss" or "time-cap" in notes,
        # or where we can compute a loss from matching buy/sell prices)
        losses = 0
        total_loss = 0.0
        for sell in sells:
            notes = (sell.notes or "").lower()
            is_loss = "stop-loss" in notes or "time-cap" in notes

            if not is_loss and buys:
                matching_buy = next(
                    (b for b in buys if b.created_at and sell.created_at
                     and b.created_at < sell.created_at),
                    None,
                )
                if matching_buy and sell.price < matching_buy.price:
                    is_loss = True

            if is_loss:
                losses += 1
                if buys:
                    entry = next(
                        (b for b in buys if b.created_at and sell.created_at
                         and b.created_at < sell.created_at),
                        None,
                    )
                    if entry:
                        total_loss += (entry.price - sell.price) * sell.size + (sell.fee or 0)

        if losses == 0:
            return False, ""

        round_trips = len(sells)

        if losses >= 3:
            self._add_cooldown(product_id, f"{losses} losses in 3h (${total_loss:.2f})", 120)
            return True, f"{losses} losses in 3h, total ${total_loss:.2f} lost"
        elif losses >= 2:
            self._add_cooldown(product_id, f"{losses} losses in 3h (${total_loss:.2f})", 60)
            return True, f"{losses} losses in 3h, total ${total_loss:.2f} lost"
        elif losses == 1 and round_trips == 1:
            # Only 1 trade and it was a loss — apply standard cooldown
            self._add_cooldown(product_id, f"1 loss (${total_loss:.2f})", _COOLDOWN_MINUTES)
            return True, f"recent loss ${total_loss:.2f}"

        return False, ""

    # ------------------------------------------------------------------
    # Smart entry analysis
    # ------------------------------------------------------------------

    def _analyze_entry(
        self, product_id: str, current_price: float, hourly_change_pct: float = 0.0,
    ) -> _EntryAnalysis:
        """Fetch intraday candles and decide if now is a good time to enter."""
        now = int(time.time())
        start = now - 14400  # 4 hours of context

        candles = []
        try:
            candles = self.client.get_candles(product_id, start, now, granularity="FIVE_MINUTE")
            candles = sorted(candles, key=lambda c: c.timestamp)
        except Exception as exc:
            logger.debug("Entry analysis: cannot fetch candles for %s: %s", product_id, exc)

        if len(candles) < 6:
            return _EntryAnalysis(
                product_id=product_id, current_price=current_price,
                range_position=0.5, pullback_pct=0.0, short_trend=0.0,
                rsi=50.0, volatility=0.0,
                intraday_high=current_price, intraday_low=current_price,
                volume_ratio=0.0,
                signal="wait", reason="insufficient candle data, waiting",
            )

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        intraday_high = max(highs)
        intraday_low = min(lows)
        price_range = intraday_high - intraday_low

        # --- Range position ---
        range_position = (current_price - intraday_low) / price_range if price_range > 0 else 0.5

        # --- Pullback from high ---
        pullback_pct = (intraday_high - current_price) / intraday_high if intraday_high > 0 else 0.0

        # --- Short-term trend (slope of last 6 candles, ~30min) ---
        recent = closes[-6:]
        if len(recent) >= 2 and recent[0] > 0:
            short_trend = (recent[-1] - recent[0]) / recent[0]
        else:
            short_trend = 0.0

        # --- RSI (14-period) ---
        rsi = self._compute_rsi(closes, period=14)

        # --- Volatility (std-dev of 5-min returns) ---
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
        if returns:
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
        else:
            volatility = 0.0

        # --- Volume ratio (recent 6 candles vs full window average) ---
        volumes = [c.volume for c in candles]
        avg_vol = sum(volumes) / len(volumes) if volumes else 1.0
        recent_vol = sum(volumes[-6:]) / min(6, len(volumes[-6:])) if volumes else 0.0
        volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 0.0

        # --- Decision logic ---
        signal, reason = self._entry_decision(
            range_position, pullback_pct, short_trend, rsi, volatility,
            hourly_change_pct=hourly_change_pct,
            volume_ratio=volume_ratio,
        )

        return _EntryAnalysis(
            product_id=product_id, current_price=current_price,
            range_position=range_position, pullback_pct=pullback_pct,
            short_trend=short_trend, rsi=rsi, volatility=volatility,
            intraday_high=intraday_high, intraday_low=intraday_low,
            volume_ratio=volume_ratio,
            signal=signal, reason=reason,
        )

    @staticmethod
    def _compute_rsi(closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        # Use exponential-weighted (Wilder's smoothing) over last `period` values
        recent_gains = gains[-(period):]
        recent_losses = losses[-(period):]
        avg_gain = sum(recent_gains) / period
        avg_loss = sum(recent_losses) / period
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _entry_decision(
        range_pos: float, pullback: float, trend: float, rsi: float, vol: float,
        hourly_change_pct: float = 0.0,
        volume_ratio: float = 1.0,
    ) -> tuple[str, str]:
        """Return (signal, reason) based on combined indicators.

        Momentum candidates are coins already moving up 5%+/hour.  The
        decision should lean towards entering because that is the whole
        point of the momentum strategy — catch moves in progress.
        Overly conservative filters cause most candidates to be "waited"
        on while they keep running, tanking decision confidence.
        """
        # Hard skip: extremely overbought on every metric
        if rsi > 80 and range_pos > 0.90:
            return "skip", f"overbought (RSI={rsi:.0f}, range={range_pos:.0%})"

        # Very large hourly move (>20%): defer to pullback to avoid buying the top
        if hourly_change_pct >= 0.20:
            return "pullback_watch", (
                f"very large move — watching for pullback (hourly=+{hourly_change_pct:.1%})"
            )

        # Volume check: only skip if volume is very low
        if volume_ratio < 0.5:
            return "wait", f"very low volume (vol_ratio={volume_ratio:.1f}x)"

        # --- Entry conditions (ordered from strongest to weakest) ---

        # Pullback entry: dipped from high and still in lower range
        if pullback >= 0.03 and range_pos < 0.70:
            return "enter", f"pullback {pullback:.1%} from high, range={range_pos:.0%}"

        # Trend-confirmed entry: positive trend with reasonable range
        if trend > 0 and range_pos < 0.75 and rsi < 70:
            return "enter", f"trend-confirmed (range={range_pos:.0%}, RSI={rsi:.0f}, trend=+{trend:.1%})"

        # Strong mover with positive trend: enter, don't wait
        if hourly_change_pct >= 0.08 and trend > 0 and rsi < 75:
            return "enter", f"strong momentum entry (hourly=+{hourly_change_pct:.1%}, RSI={rsi:.0f})"

        # Mid-range entry: moderate position with any positive trend
        if 0.20 <= range_pos <= 0.65 and trend > 0:
            return "enter", f"mid-range entry (range={range_pos:.0%}, trend=+{trend:.1%})"

        # High in range but with pullback underway: watch for re-entry
        if range_pos > 0.75 and pullback < 0.02 and rsi > 65:
            return "pullback_watch", (
                f"near peak — watching for pullback (range={range_pos:.0%}, RSI={rsi:.0f})"
            )

        # Negative trend: wait briefly
        if trend <= -0.005:
            return "wait", f"negative trend ({trend:+.1%}), waiting for upturn"

        # Default: enter with a note (momentum candidates deserve the benefit of the doubt)
        if trend >= 0:
            return "enter", f"momentum entry (range={range_pos:.0%}, RSI={rsi:.0f}, trend={trend:+.1%})"

        return "wait", f"weak setup (range={range_pos:.0%}, RSI={rsi:.0f}, trend={trend:+.1%})"

    # ------------------------------------------------------------------
    # Candidate report (rich per-coin log)
    # ------------------------------------------------------------------

    def _build_candidate_report(
        self, product_id: str, current_price: float,
        hourly_change: float, analysis: _EntryAnalysis,
    ) -> str:
        """Build a detailed one-line report for a momentum candidate."""
        parts: list[str] = []

        # --- Ideal entry & exit ---
        sl, tp = self._compute_dynamic_levels(current_price, analysis)
        ideal_entry = analysis.intraday_low + (analysis.intraday_high - analysis.intraday_low) * 0.35
        parts.append(
            f"now=${self._format_price(current_price, product_id)} "
            f"ideal_in=${self._format_price(ideal_entry, product_id)} "
            f"target=${self._format_price(tp, product_id)}(+{(tp/current_price-1)*100:.1f}%) "
            f"stop=${self._format_price(sl, product_id)}(-{(1-sl/current_price)*100:.1f}%)"
        )

        # --- Intraday range ---
        parts.append(
            f"4h_range=${self._format_price(analysis.intraday_low, product_id)}"
            f"-${self._format_price(analysis.intraday_high, product_id)}"
        )

        # --- Technical indicators ---
        parts.append(
            f"RSI={analysis.rsi:.0f} "
            f"range_pos={analysis.range_position:.0%} "
            f"trend={analysis.short_trend:+.1%} "
            f"vol={analysis.volume_ratio:.1f}x"
        )

        # --- Previous trade history ---
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)
        trades = self.db.get_recent_trades_for(product_id, since, strategy=STRATEGY)
        if trades:
            buys = [t for t in trades if t.side == "BUY" and t.status == "FILLED"]
            sells = [t for t in trades if t.side == "SELL" and t.status == "FILLED"]
            wins = 0
            losses_count = 0
            net_pnl = 0.0
            for sell in sells:
                matching = next(
                    (b for b in buys if b.created_at and sell.created_at
                     and b.created_at < sell.created_at),
                    None,
                )
                if matching:
                    pnl = (sell.price - matching.price) * sell.size - (sell.fee or 0)
                    net_pnl += pnl
                    if pnl >= 0:
                        wins += 1
                    else:
                        losses_count += 1

            # Post-buy behavior: for unfilled buys still open, how did price move?
            post_buy_notes: list[str] = []
            for buy in buys:
                if not buy.created_at:
                    continue
                was_sold = any(
                    s for s in sells
                    if s.created_at and s.created_at > buy.created_at
                )
                move_pct = (current_price - buy.price) / buy.price * 100 if buy.price > 0 else 0
                if was_sold:
                    sell_match = next(
                        (s for s in sells if s.created_at and s.created_at > buy.created_at),
                        None,
                    )
                    if sell_match:
                        exit_pct = (sell_match.price - buy.price) / buy.price * 100
                        post_buy_notes.append(f"bought@${self._format_price(buy.price, product_id)}→exit{exit_pct:+.1f}%")
                else:
                    post_buy_notes.append(f"bought@${self._format_price(buy.price, product_id)}→now{move_pct:+.1f}%")

            history_str = f"history: {len(buys)}buys {wins}W/{losses_count}L net=${net_pnl:+.2f}"
            if post_buy_notes:
                history_str += " [" + ", ".join(post_buy_notes[-3:]) + "]"
            parts.append(history_str)
        else:
            parts.append("history: first trade")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Historical seed
    # ------------------------------------------------------------------

    def _seed_price_buffer(self) -> None:
        """Pre-fill the price buffer with ~2 hours of recent candle data
        so momentum detection works immediately on start."""
        now = int(time.time())
        start = now - 7200  # 2 hours ago
        seeded = 0
        for pid in self.coin_ids:
            try:
                candles = self.client.get_candles(pid, start, now, granularity="FIVE_MINUTE")
                if not candles:
                    continue
                buf: deque[_PriceEntry] = deque(maxlen=self._max_history)
                for c in sorted(candles, key=lambda x: x.timestamp):
                    buf.append(_PriceEntry(price=c.close, ts=c.timestamp))
                self._prices[pid] = buf
                seeded += 1
            except Exception:
                pass
        logger.info("Momentum: seeded price buffer for %d/%d coins (2h history)", seeded, len(self.coin_ids))

    # ------------------------------------------------------------------
    # Scan loop (new entries)
    # ------------------------------------------------------------------

    def _scan_loop(self) -> None:
        while not self._scan_stop.is_set():
            try:
                self._scan_and_enter()
            except Exception as e:
                logger.error("Momentum scan error: %s", e)
            for _ in range(self.cfg.scan_interval_sec):
                if self._scan_stop.is_set():
                    break
                if MomentumScanner.consume_scan_trigger():
                    logger.info("Force-scan triggered via dashboard")
                    break
                self._scan_stop.wait(timeout=1)

    def _scan_and_enter(self) -> None:
        if self.risk.state in (
            TradingState.EMERGENCY,
            TradingState.SHUTDOWN,
            TradingState.HALTED_DAILY,
            TradingState.HALTED_HOURLY,
        ):
            return

        now = time.time()

        # Fetch current prices for all coins
        current_prices: dict[str, float] = {}
        for pid in self.coin_ids:
            if self._scan_stop.is_set():
                return
            try:
                current_prices[pid] = self.client.get_ticker(pid)
            except Exception:
                pass

        # Record prices in rolling buffer
        for pid, price in current_prices.items():
            if pid not in self._prices:
                self._prices[pid] = deque(maxlen=self._max_history)
            self._prices[pid].append(_PriceEntry(price=price, ts=now))

        # Detect momentum candidates (exclude blocked coins)
        candidates = [
            (pid, pct) for pid, pct in self._detect_candidates(current_prices, now)
            if not self._is_blocked(pid)
        ]

        # Merge new candidates into watchlist (skip those already in pullback-watch)
        for pid, pct in candidates:
            if pid in self._pullback_watches:
                continue
            if pid not in self._watchlist:
                self._watchlist[pid] = _WatchlistEntry(
                    product_id=pid,
                    hourly_change_pct=pct,
                    added_at=now,
                    expiry=now + _WATCHLIST_TTL_SEC,
                )
            else:
                self._watchlist[pid].hourly_change_pct = pct

        # Expire stale watchlist entries
        expired = [pid for pid, w in self._watchlist.items() if now > w.expiry]
        for pid in expired:
            logger.debug("Momentum: %s watchlist expired (no good entry found)", pid)
            del self._watchlist[pid]

        # Also remove candidates that dropped below threshold
        active_pids = {pid for pid, _ in candidates}
        dropped = [pid for pid in list(self._watchlist) if pid not in active_pids]
        for pid in dropped:
            del self._watchlist[pid]

        with self._lock:
            self._stats.last_scan_time = dt.datetime.now(dt.timezone.utc).isoformat()
            self._stats.coins_scanned = len(current_prices)
            self._stats.candidates_found = len(candidates)
        self._persist_scan_stats()

        open_mom = len(self.db.get_open_positions(strategy=STRATEGY))
        if candidates:
            top = ", ".join(f"{pid}(+{pct*100:.1f}%)" for pid, pct in candidates[:5])
            logger.info(
                "Momentum scan: %d coins, %d candidates [%s], %d watching, %d open",
                len(current_prices), len(candidates), top, len(self._watchlist), open_mom,
            )
        else:
            logger.debug("Momentum scan: %d coins, 0 candidates, %d watching, %d open",
                         len(current_prices), len(self._watchlist), open_mom)

        if not self._watchlist:
            return

        # Enter positions from the watchlist (respecting limits)
        open_momentum = self.db.get_open_positions(strategy=STRATEGY)
        open_pids = {p.product_id for p in open_momentum}
        slots_available = self.cfg.max_open_positions - len(open_momentum)

        if slots_available <= 0:
            return

        ml_pids = {p.product_id for p in self.db.get_open_positions(strategy="ml")}
        portfolio_value = self.client.get_portfolio_value(current_prices)
        position_budget = portfolio_value * self.cfg.max_position_pct
        usdc_available = self.client.get_usd_balance()

        # Sort watchlist by hourly change (strongest first)
        sorted_watch = sorted(
            self._watchlist.values(), key=lambda w: w.hourly_change_pct, reverse=True
        )

        entered = 0
        to_remove: list[str] = []

        for watch in sorted_watch:
            pid = watch.product_id
            if entered >= slots_available:
                break
            if pid in open_pids or pid in ml_pids:
                to_remove.append(pid)
                continue
            if self._is_blocked(pid):
                to_remove.append(pid)
                continue
            if self._scan_stop.is_set():
                return

            # Check recent trade history — don't repeat losses
            skip, loss_reason = self._check_recent_losses(pid)
            if skip:
                logger.info("Momentum SKIP %s (+%.1f%%): %s", pid, watch.hourly_change_pct * 100, loss_reason)
                to_remove.append(pid)
                continue

            price = current_prices.get(pid)
            if not price or price <= 0:
                continue

            # Scale-in: large moves start with 25% of budget
            is_scale_in = watch.hourly_change_pct >= self.cfg.scale_in_min_change_pct
            if is_scale_in:
                size_usd = min(position_budget * self.cfg.scale_in_initial_pct, usdc_available)
            else:
                size_usd = min(position_budget, usdc_available)
            if size_usd < 1.0:
                logger.debug("Momentum: insufficient funds ($%.2f), stopping entries", usdc_available)
                break

            # Analyze entry timing
            analysis = self._analyze_entry(pid, price, watch.hourly_change_pct)
            report = self._build_candidate_report(pid, price, watch.hourly_change_pct, analysis)

            if analysis.signal == "skip":
                logger.info("Momentum SKIP %s (+%.1f%%): %s | %s",
                            pid, watch.hourly_change_pct * 100, analysis.reason, report)
                self._log_decision(pid, "skip", price, watch.hourly_change_pct, analysis)
                to_remove.append(pid)
                continue

            if analysis.signal == "wait":
                logger.info("Momentum WAIT %s (+%.1f%%): %s | %s",
                            pid, watch.hourly_change_pct * 100, analysis.reason, report)
                self._log_decision(pid, "wait", price, watch.hourly_change_pct, analysis)
                continue

            if analysis.signal == "pullback_watch":
                if pid not in self._pullback_watches:
                    now_ts = time.time()
                    self._pullback_watches[pid] = _PullbackWatch(
                        product_id=pid,
                        hourly_change_pct=watch.hourly_change_pct,
                        peak_price=price,
                        local_low=price,
                        phase="pullback",
                        added_at=now_ts,
                        expiry=now_ts + self.cfg.pullback_watch_ttl_sec,
                    )
                    logger.info(
                        "Momentum PULLBACK-WATCH %s (+%.1f%%) @ $%.6g: %s | %s",
                        pid, watch.hourly_change_pct * 100, price,
                        analysis.reason, report,
                    )
                self._log_decision(pid, "pullback_watch", price, watch.hourly_change_pct, analysis)
                to_remove.append(pid)
                continue

            # signal == "enter"
            entry_mode = "SCALE-IN 1/6" if is_scale_in else "FULL"
            logger.info("Momentum ENTER %s (+%.1f%%) [%s]: %s | %s",
                        pid, watch.hourly_change_pct * 100, entry_mode, analysis.reason, report)
            self._log_decision(pid, "enter", price, watch.hourly_change_pct, analysis)

            ok = self._enter_momentum(
                pid, price, size_usd, watch.hourly_change_pct, analysis
            )
            if ok:
                usdc_available -= size_usd
                entered += 1
                to_remove.append(pid)
                if is_scale_in:
                    with self._lock:
                        self._scale_in[pid] = _ScaleInState(
                            product_id=pid,
                            base_budget=position_budget,
                            total_invested_usd=size_usd,
                            tranches_done=1,
                            last_tranche_time=time.time(),
                        )

        for pid in to_remove:
            self._watchlist.pop(pid, None)

        self._persist_watchlist()

        self._check_pullback_entries(current_prices)
        self._check_scale_in_adds(current_prices)

    # ------------------------------------------------------------------
    # Pullback re-entry: wait for correction + recovery on strong movers
    # ------------------------------------------------------------------

    def _check_pullback_entries(self, current_prices: dict[str, float]) -> None:
        """Monitor pullback-watch coins: update peak/low, detect recovery, enter."""
        if not self._pullback_watches:
            return

        now = time.time()
        cfg = self.cfg
        to_remove: list[str] = []

        open_pids = {p.product_id for p in self.db.get_open_positions(strategy=STRATEGY)}

        for pid, pw in list(self._pullback_watches.items()):
            if self._scan_stop.is_set():
                return

            # Expired
            if now > pw.expiry:
                logger.info("Pullback-watch EXPIRED %s after %.0fmin — no re-entry found",
                            pid, (now - pw.added_at) / 60)
                to_remove.append(pid)
                continue

            # Already have a position
            if pid in open_pids or self._is_blocked(pid):
                to_remove.append(pid)
                continue

            price = current_prices.get(pid)
            if not price or price <= 0:
                continue

            # Update peak (price may still be rising)
            if price > pw.peak_price:
                pw.peak_price = price

            # Update local low
            if price < pw.local_low:
                pw.local_low = price

            drop_from_peak = (pw.peak_price - price) / pw.peak_price if pw.peak_price > 0 else 0
            bounce_from_low = (price - pw.local_low) / pw.local_low if pw.local_low > 0 else 0

            if pw.phase == "pullback":
                if drop_from_peak >= cfg.pullback_min_drop_pct:
                    pw.phase = "recovery"
                    pw.local_low = price
                    logger.info(
                        "Pullback-watch %s: pullback %.1f%% from peak $%.6g → "
                        "now watching for recovery (low=$%.6g)",
                        pid, drop_from_peak * 100, pw.peak_price, price,
                    )
                else:
                    logger.debug(
                        "Pullback-watch %s: phase=pullback, drop=%.1f%% (need %.0f%%)",
                        pid, drop_from_peak * 100, cfg.pullback_min_drop_pct * 100,
                    )

            if pw.phase == "recovery":
                # Re-analyze short-term trend via candles
                analysis = self._analyze_entry(pid, price, pw.hourly_change_pct)
                trend_positive = analysis.short_trend > 0

                if bounce_from_low >= cfg.pullback_min_bounce_pct and trend_positive:
                    logger.info(
                        "Pullback-watch ENTER %s: bounce +%.1f%% off low $%.6g, "
                        "trend=+%.2f%%, peak was $%.6g (drop was %.1f%%)",
                        pid, bounce_from_low * 100, pw.local_low,
                        analysis.short_trend * 100, pw.peak_price, drop_from_peak * 100,
                    )

                    portfolio_value = self.client.get_portfolio_value(current_prices)
                    position_budget = portfolio_value * cfg.max_position_pct

                    is_scale_in = pw.hourly_change_pct >= cfg.scale_in_min_change_pct
                    if is_scale_in:
                        size_usd = position_budget * cfg.scale_in_initial_pct
                    else:
                        size_usd = position_budget

                    usdc_available = self.client.get_usd_balance()
                    size_usd = min(size_usd, usdc_available)
                    if size_usd < 1.0:
                        logger.debug("Pullback re-entry: insufficient funds ($%.2f)", usdc_available)
                        continue

                    self._log_decision(pid, "enter", price, pw.hourly_change_pct, analysis)
                    ok = self._enter_momentum(pid, price, size_usd, pw.hourly_change_pct, analysis)
                    if ok:
                        to_remove.append(pid)
                        if is_scale_in:
                            with self._lock:
                                self._scale_in[pid] = _ScaleInState(
                                    product_id=pid,
                                    base_budget=position_budget,
                                    total_invested_usd=size_usd,
                                    tranches_done=1,
                                    last_tranche_time=time.time(),
                                )
                else:
                    logger.debug(
                        "Pullback-watch %s: phase=recovery, bounce=%.1f%% (need %.1f%%), "
                        "trend=%s",
                        pid, bounce_from_low * 100, cfg.pullback_min_bounce_pct * 100,
                        "positive" if trend_positive else "negative",
                    )

        for pid in to_remove:
            self._pullback_watches.pop(pid, None)

    # ------------------------------------------------------------------
    # Scale-in: gradual position building for large moves
    # ------------------------------------------------------------------

    def _scale_in_tranche_interval(self) -> float:
        """Seconds between scale-in tranches, evenly spaced over max_hold_minutes."""
        num_adds = (self.cfg.scale_in_max_pct - self.cfg.scale_in_initial_pct) / self.cfg.scale_in_step_pct
        return (self.cfg.max_hold_minutes * 60) / max(num_adds, 1)

    def _check_scale_in_adds(self, current_prices: dict[str, float]) -> None:
        """Check all active scale-in positions and add tranches when due."""
        with self._lock:
            pids = list(self._scale_in.keys())
        if not pids:
            return

        now = time.time()
        interval = self._scale_in_tranche_interval()
        completed: list[str] = []

        for pid in pids:
            with self._lock:
                state = self._scale_in.get(pid)
            if not state:
                continue

            # Already at max?
            max_total = state.base_budget * self.cfg.scale_in_max_pct
            if state.total_invested_usd >= max_total:
                completed.append(pid)
                continue

            # Too early for next tranche?
            if now - state.last_tranche_time < interval:
                continue

            # Position still open?
            pos = self.db.get_position(pid)
            if not pos or not pos.is_open:
                completed.append(pid)
                continue

            price = current_prices.get(pid)
            if not price or price <= 0:
                try:
                    price = self.client.get_ticker(pid)
                except Exception:
                    continue

            # Condition: position must be profitable
            if price <= pos.entry_price:
                logger.debug("Scale-in SKIP %s: not profitable (now=$%.4f <= entry=$%.4f)",
                             pid, price, pos.entry_price)
                continue

            # Condition: short-term trend must be positive
            analysis = self._analyze_entry(pid, price)
            if analysis.short_trend <= 0:
                logger.debug("Scale-in SKIP %s: trend negative (%.2f%%)", pid, analysis.short_trend * 100)
                continue

            tranche_usd = min(
                state.base_budget * self.cfg.scale_in_step_pct,
                max_total - state.total_invested_usd,
            )
            if tranche_usd < 1.0:
                completed.append(pid)
                continue

            max_tranches = int(self.cfg.scale_in_max_pct / self.cfg.scale_in_step_pct)
            tranche_num = state.tranches_done + 1
            logger.info(
                "MOMENTUM SCALE-IN %s: tranche %d/%d, +$%.2f (total $%.2f / $%.2f) price=$%s trend=%+.1f%%",
                pid, tranche_num, max_tranches, tranche_usd,
                state.total_invested_usd + tranche_usd, max_total,
                self._format_price(price, pid), analysis.short_trend * 100,
            )

            ok = self._enter_scale_in_tranche(pid, price, tranche_usd, pos)
            if ok:
                with self._lock:
                    st = self._scale_in.get(pid)
                    if st:
                        st.total_invested_usd += tranche_usd
                        st.tranches_done += 1
                        st.last_tranche_time = now
                        if st.total_invested_usd >= max_total:
                            completed.append(pid)

        for pid in completed:
            with self._lock:
                self._scale_in.pop(pid, None)

    def _enter_scale_in_tranche(
        self, product_id: str, price: float, tranche_usd: float, existing_pos: Position
    ) -> bool:
        """Buy an additional tranche and update the existing position."""
        current_positions = [
            {"product_id": p.product_id, "is_open": True, "value_usd": p.size * price}
            for p in self.db.get_open_positions()
        ]
        check = self.risk.validate_trade(
            product_id=product_id,
            side="BUY",
            size_usd=tranche_usd,
            portfolio_value=self.client.get_portfolio_value(),
            current_positions=current_positions,
            coin_universe=self.coin_ids,
        )
        if not check.allowed:
            logger.info("Scale-in BUY rejected %s: %s", product_id, check.reason)
            return False

        tranche_usd = check.max_size_usd

        if self.fx:
            if not self.fx.ensure_usdc_for_trade(tranche_usd):
                logger.info("Scale-in: could not secure USDC for %s ($%.2f)", product_id, tranche_usd)
                return False

        round_trip_fee = self.taker_fee * 2
        effective_usd = tranche_usd / (1 + round_trip_fee)
        buy_fee_est = effective_usd * self.taker_fee

        add_size = effective_usd / price
        quote_size = f"{effective_usd:.2f}"

        try:
            result = self.client.place_market_buy(
                product_id=product_id,
                quote_size=quote_size,
            )
        except Exception as e:
            logger.error("Scale-in buy failed %s: %s", product_id, e)
            return False

        if result.status == "FAILED":
            logger.warning("Scale-in buy rejected %s: %s", product_id, result.raw)
            return False

        order_id = result.order_id or f"mom-si-{uuid.uuid4()}"

        actual_fee = buy_fee_est
        actual_size = add_size
        actual_price = price
        try:
            order_data = self.client.get_order(order_id).get("order", {})
            if order_data.get("status") in ("FILLED", "COMPLETED"):
                actual_fee = self._extract_fee(order_data, buy_fee_est)
                filled_size = float(order_data.get("filled_size") or 0)
                filled_value = float(order_data.get("filled_value") or 0)
                if filled_size > 0:
                    actual_size = filled_size
                    actual_price = filled_value / filled_size if filled_value > 0 else price
        except Exception:
            pass

        trade = Trade(
            order_id=order_id,
            product_id=product_id,
            side="BUY",
            size=actual_size,
            price=actual_price,
            fee=actual_fee,
            status="FILLED",
            signal_strength=0,
            strategy=STRATEGY,
            notes=f"momentum scale-in tranche +${tranche_usd:.2f}",
        )
        self.db.save_trade(trade)

        old_size = existing_pos.size
        old_entry = existing_pos.entry_price
        new_total_size = old_size + actual_size
        new_entry_price = (old_entry * old_size + actual_price * actual_size) / new_total_size

        self.db.update_position(
            product_id,
            size=new_total_size,
            entry_price=new_entry_price,
        )

        with self._lock:
            self._stats.total_trades += 1
        logger.info(
            "SCALE-IN FILLED %s: size %.6f→%.6f, avg_entry $%s→$%s",
            product_id, old_size, new_total_size,
            self._format_price(old_entry, product_id),
            self._format_price(new_entry_price, product_id),
        )
        return True

    # ------------------------------------------------------------------
    # Monitor loop (exit checks — runs independently of scanner)
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        logger.info("Momentum monitor started (checking exits every %ds)", self.cfg.scan_interval_sec)
        while not self._monitor_stop.is_set():
            try:
                positions = self.db.get_open_positions(strategy=STRATEGY)
                has_scale_in = bool(self._scale_in)
                has_pullback = bool(self._pullback_watches)
                if not positions and not self.is_running and not has_scale_in and not has_pullback:
                    logger.info("Momentum monitor: no open positions and scanner stopped, exiting")
                    break
                if positions:
                    self._check_momentum_exits(time.time())
                # Continue scale-in / pullback watches even when the scanner is stopped
                if (has_scale_in or has_pullback) and not self.is_running:
                    current_prices: dict[str, float] = {}
                    with self._lock:
                        pids_to_fetch = list(self._scale_in.keys()) + list(self._pullback_watches.keys())
                    for pid in pids_to_fetch:
                        try:
                            current_prices[pid] = self.client.get_ticker(pid)
                        except Exception:
                            pass
                    if current_prices:
                        if has_pullback:
                            self._check_pullback_entries(current_prices)
                        if has_scale_in:
                            self._check_scale_in_adds(current_prices)
                # Track decision outcomes for confidence scoring
                if self._decision_log:
                    self._measure_decision_outcomes()
            except Exception as e:
                logger.error("Momentum monitor error: %s", e)
            self._monitor_stop.wait(timeout=self.cfg.scan_interval_sec)

    # ------------------------------------------------------------------
    # Candidate detection
    # ------------------------------------------------------------------

    def _detect_candidates(
        self, current_prices: dict[str, float], now: float
    ) -> list[tuple[str, float]]:
        """Return (product_id, change_pct) pairs sorted by strength (descending)."""
        one_hour_ago = now - 3600
        candidates: list[tuple[str, float]] = []

        for pid, price_now in current_prices.items():
            if price_now <= 0:
                continue
            buf = self._prices.get(pid)
            if not buf:
                continue

            # Find the oldest price at least ~1 hour ago
            price_then = None
            for entry in buf:
                if entry.ts <= one_hour_ago:
                    price_then = entry.price
                    break
            if price_then is None or price_then <= 0:
                continue

            change_pct = (price_now - price_then) / price_then
            if change_pct >= self.cfg.min_hourly_change_pct and change_pct < 0.50:
                candidates.append((pid, change_pct))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _compute_dynamic_levels(
        self, price: float, analysis: _EntryAnalysis | None
    ) -> tuple[float, float]:
        """Compute stop-loss and take-profit using intraday volatility."""
        if analysis is None or analysis.volatility <= 0:
            return (
                price * (1 - self.cfg.hard_stop_pct),
                price * (1 + self.cfg.take_profit_pct),
            )

        vol = analysis.volatility
        intraday_range = analysis.intraday_high - analysis.intraday_low
        range_pct = intraday_range / price if price > 0 else 0

        # Stop-loss: use the larger of config hard_stop or 1.5x volatility,
        # but floor at the intraday low minus a small buffer
        sl_pct = max(self.cfg.hard_stop_pct, vol * 30)  # vol is per-5min, scale up
        sl_pct = min(sl_pct, 0.08)  # cap at 8%
        stop_loss = price * (1 - sl_pct)
        # Don't set SL above intraday low (would trigger immediately on normal swings)
        if analysis.intraday_low > 0:
            floor = analysis.intraday_low * 0.995
            stop_loss = min(stop_loss, floor)

        # Take-profit: scale with the surge magnitude
        # Bigger surges → bigger TP targets
        tp_pct = max(self.cfg.take_profit_pct, range_pct * 0.5)
        tp_pct = min(tp_pct, 0.15)  # cap at 15%
        take_profit = price * (1 + tp_pct)

        return stop_loss, take_profit

    def _enter_momentum(
        self, product_id: str, price: float, size_usd: float,
        change_pct: float, analysis: _EntryAnalysis | None = None,
    ) -> bool:
        """Attempt to enter a momentum position. Returns True on success."""
        if price <= 0 or size_usd <= 0:
            return False

        # Validate through risk manager
        current_positions = [
            {"product_id": p.product_id, "is_open": True, "value_usd": p.size * price}
            for p in self.db.get_open_positions()
        ]
        check = self.risk.validate_trade(
            product_id=product_id,
            side="BUY",
            size_usd=size_usd,
            portfolio_value=self.client.get_portfolio_value(),
            current_positions=current_positions,
            coin_universe=self.coin_ids,
        )
        if not check.allowed:
            logger.info("Momentum BUY rejected %s: %s", product_id, check.reason)
            return False

        size_usd = check.max_size_usd

        # Ensure USDC available
        if self.fx:
            if not self.fx.ensure_usdc_for_trade(size_usd):
                logger.info("Momentum: could not secure USDC for %s ($%.2f)", product_id, size_usd)
                return False

        # Reserve fees (use taker fee since we're placing market orders)
        round_trip_fee = self.taker_fee * 2
        effective_usd = size_usd / (1 + round_trip_fee)
        buy_fee_est = effective_usd * self.taker_fee

        size_base = effective_usd / price
        quote_size = f"{effective_usd:.2f}"

        logger.info(
            "MOMENTUM BUY %s: ~%.1f @ $%s ($%.2f) hourly_change=+%.1f%%",
            product_id, size_base, self._format_price(price, product_id),
            effective_usd, change_pct * 100,
        )

        try:
            result = self.client.place_market_buy(
                product_id=product_id,
                quote_size=quote_size,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if any(ue in err_msg for ue in _UNTRADEABLE_ERRORS):
                self._blacklist.add(product_id)
                logger.warning("Momentum: blacklisted %s (untradeable: %s)", product_id, e)
            else:
                logger.error("Momentum buy failed %s: %s", product_id, e)
            return False

        if result.status == "FAILED":
            raw_str = str(result.raw).lower() if result.raw else ""
            if any(ue in raw_str for ue in _UNTRADEABLE_ERRORS):
                self._blacklist.add(product_id)
                logger.warning("Momentum: blacklisted %s (untradeable)", product_id)
            elif "insufficient" in raw_str:
                logger.warning("Momentum buy rejected %s: insufficient funds", product_id)
            else:
                logger.warning("Momentum buy rejected %s: %s", product_id, result.raw)
            return False

        order_id = result.order_id or f"mom-{uuid.uuid4()}"

        # Market orders fill immediately; extract actual fill details
        actual_fee = buy_fee_est
        actual_size = size_base
        actual_price = price
        try:
            order_data = self.client.get_order(order_id).get("order", {})
            if order_data.get("status") in ("FILLED", "COMPLETED"):
                actual_fee = self._extract_fee(order_data, buy_fee_est)
                filled_value = float(order_data.get("filled_value") or 0)
                filled_size = float(order_data.get("filled_size") or 0)
                if filled_size > 0:
                    actual_size = filled_size
                    actual_price = filled_value / filled_size if filled_value > 0 else price
        except Exception:
            pass

        trade = Trade(
            order_id=order_id,
            product_id=product_id,
            side="BUY",
            size=actual_size,
            price=actual_price,
            fee=actual_fee,
            status="FILLED",
            signal_strength=change_pct,
            strategy=STRATEGY,
            notes=f"momentum +{change_pct*100:.1f}%",
        )
        self.db.save_trade(trade)

        stop_loss, take_profit = self._compute_dynamic_levels(actual_price, analysis)

        position = Position(
            product_id=product_id,
            side="LONG",
            entry_price=actual_price,
            size=actual_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=actual_price,
            strategy=STRATEGY,
        )
        self.db.save_position(position)
        with self._lock:
            self._stats.total_trades += 1
        logger.info(
            "MOMENTUM FILLED %s: SL=$%s (%.1f%%) TP=$%s (+%.1f%%) hold_max=%dmin",
            product_id,
            self._format_price(stop_loss, product_id),
            (1 - stop_loss / actual_price) * 100,
            self._format_price(take_profit, product_id),
            (take_profit / actual_price - 1) * 100,
            self.cfg.max_hold_minutes,
        )
        return True

    # ------------------------------------------------------------------
    # Exit logic (trailing stop / take profit / time cap / hard stop)
    # ------------------------------------------------------------------

    def _check_momentum_exits(self, now: float) -> None:
        positions = self.db.get_open_positions(strategy=STRATEGY)
        if not positions:
            return

        for pos in positions:
            if self._monitor_stop.is_set():
                return
            try:
                current_price = self.client.get_ticker(pos.product_id)
            except Exception:
                continue

            if current_price <= 0:
                continue

            age_secs = 0.0
            if pos.opened_at:
                age_secs = (dt.datetime.utcnow() - pos.opened_at).total_seconds()

            in_eval_hold = age_secs < self.cfg.evaluation_hold_sec

            # During evaluation hold: only allow catastrophic stop (e.g. -10%)
            # After evaluation hold: normal exit management
            if in_eval_hold:
                catastrophic_price = pos.entry_price * (1 - self.cfg.catastrophic_stop_pct)
                if current_price <= catastrophic_price:
                    exit_reason = f"catastrophic-stop (eval hold, {self.cfg.catastrophic_stop_pct:.0%} drop)"
                    is_loss = True
                    self._exit_momentum(pos, current_price, exit_reason)
                    self._add_cooldown(pos.product_id, exit_reason)
                else:
                    # Still track highest price during eval hold
                    if current_price > (pos.highest_price or 0):
                        self.db.update_position(pos.product_id, highest_price=current_price)
                continue

            # Update highest price and trailing stop (post eval-hold)
            if current_price > (pos.highest_price or 0):
                self.db.update_position(pos.product_id, highest_price=current_price)
                trailing_stop = current_price * (1 - self.cfg.trailing_stop_pct)
                if trailing_stop > (pos.stop_loss or 0):
                    self.db.update_position(pos.product_id, stop_loss=trailing_stop)
                    pos.stop_loss = trailing_stop

            exit_reason = None

            # Hard stop / trailing stop
            if pos.stop_loss and current_price <= pos.stop_loss:
                exit_reason = "stop-loss"

            # Take profit
            elif pos.take_profit and current_price >= pos.take_profit:
                exit_reason = "take-profit"

            # Time cap
            elif pos.opened_at:
                age_minutes = age_secs / 60
                if age_minutes >= self.cfg.max_hold_minutes:
                    exit_reason = f"time-cap ({int(age_minutes)}min)"

            if exit_reason:
                is_loss = current_price < pos.entry_price
                self._exit_momentum(pos, current_price, exit_reason)
                if is_loss:
                    self._add_cooldown(pos.product_id, exit_reason)

    def _exit_momentum(
        self, position: Position, current_price: float, reason: str
    ) -> None:
        product_id = position.product_id
        with self._lock:
            self._scale_in.pop(product_id, None)
            self._pullback_watches.pop(product_id, None)
        sell_value = current_price * position.size
        taker_fee = sell_value * self.taker_fee
        gross_pnl = (current_price - position.entry_price) * position.size
        net_pnl = gross_pnl - taker_fee

        size_str = self._format_size(position.size, product_id)

        if float(size_str) <= 0:
            logger.warning("MOMENTUM EXIT skipped %s: dust position (%.10f)", product_id, position.size)
            self.db.close_position(product_id, pnl=0)
            return

        logger.warning(
            "MOMENTUM EXIT %s (%s): gross=$%.2f fee=$%.4f net=$%.2f",
            product_id, reason, gross_pnl, taker_fee, net_pnl,
        )

        try:
            result = self.client.place_market_sell(product_id, size_str)
            order_id = result.order_id or f"mom-exit-{uuid.uuid4()}"
            trade = Trade(
                order_id=order_id,
                product_id=product_id,
                side="SELL",
                size=position.size,
                price=current_price,
                fee=taker_fee,
                status="FILLED",
                signal_strength=0,
                strategy=STRATEGY,
                notes=f"momentum exit: {reason}",
            )
            self.db.save_trade(trade)
            self.db.close_position(product_id, net_pnl)
            with self._lock:
                self._stats.total_trades += 1
            self._record_yield(position, current_price, reason, net_pnl, taker_fee)
            if self.fx:
                self.fx.rebalance_to_eurc()
        except Exception as e:
            logger.error("Momentum exit failed %s: %s", product_id, e)

    def _record_yield(
        self, position: Position, exit_price: float,
        reason: str, net_pnl: float, exit_fee: float,
    ) -> None:
        """Link a closed position back to its enter decision record."""
        entry_value = position.entry_price * position.size
        if entry_value <= 0:
            return
        pnl_pct = round(net_pnl / entry_value, 4)
        hold_sec = 0
        if position.opened_at:
            hold_sec = int((dt.datetime.utcnow() - position.opened_at).total_seconds())
        fee_pct = round((exit_fee * 2) / entry_value, 4)  # approximate round-trip

        matched = False
        for rec in reversed(self._decision_log):
            if (rec.product_id == position.product_id
                    and rec.decision == "enter"
                    and rec.realized_pnl_pct is None):
                rec.realized_pnl_pct = pnl_pct
                rec.exit_reason = reason
                rec.hold_seconds = hold_sec
                rec.entry_fee_pct = fee_pct
                matched = True
                break
        if matched:
            self._save_decision_log()

    # ------------------------------------------------------------------
    # Order / formatting helpers (reuse executor patterns)
    # ------------------------------------------------------------------

    def _get_product(self, product_id: str):
        if product_id not in self._product_cache:
            self._product_cache[product_id] = self.client.get_product(product_id)
        return self._product_cache[product_id]

    def _format_size(self, size: float, product_id: str) -> str:
        try:
            from decimal import Decimal, ROUND_DOWN
            product = self._get_product(product_id)
            inc = product.base_increment
            d_val = Decimal(str(size))
            d_inc = Decimal(str(inc))
            rounded = float((d_val / d_inc).to_integral_value(rounding=ROUND_DOWN) * d_inc)
            decimals = max(0, -Decimal(str(inc)).normalize().as_tuple().exponent)
            return f"{rounded:.{decimals}f}"
        except Exception:
            return f"{size:.8f}"

    def _format_price(self, price: float, product_id: str | None = None) -> str:
        if product_id:
            try:
                from decimal import Decimal, ROUND_DOWN
                product = self._get_product(product_id)
                inc = product.quote_increment
                d_val = Decimal(str(price))
                d_inc = Decimal(str(inc))
                rounded = float((d_val / d_inc).to_integral_value(rounding=ROUND_DOWN) * d_inc)
                decimals = max(0, -Decimal(str(inc)).normalize().as_tuple().exponent)
                return f"{rounded:.{decimals}f}"
            except Exception:
                pass
        if price >= 1.0:
            return f"{price:.2f}"
        elif price >= 0.001:
            return f"{price:.6f}"
        return f"{price:.10f}"

    def _wait_for_fill(self, order_id: str) -> tuple[bool, dict]:
        if not order_id:
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
            except Exception as exc:
                err_str = str(exc).lower()
                if "not found" in err_str or "404" in err_str:
                    not_found_count += 1
                    if not_found_count >= 3:
                        logger.info(
                            "Momentum order %s not found after %d attempts — likely filled instantly",
                            order_id, not_found_count,
                        )
                        return True, {}
                else:
                    logger.error("Momentum order check failed %s: %s", order_id, exc)
                    return False, {}
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
