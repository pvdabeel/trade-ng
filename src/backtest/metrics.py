"""Backtest performance metrics: Sharpe, drawdown, win rate, etc."""

from __future__ import annotations

import numpy as np

from src.backtest.engine import BacktestResult


def compute_metrics(result: BacktestResult) -> dict:
    """Compute performance metrics from a BacktestResult."""
    curve = np.array(result.equity_curve)
    trades = result.trades

    if len(curve) < 2:
        return _empty_metrics(result.initial_capital)

    final = curve[-1]
    total_return = (final / result.initial_capital - 1) * 100

    # Returns series
    returns = np.diff(curve) / curve[:-1]
    returns = returns[np.isfinite(returns)]

    # Sharpe ratio (annualized, assuming hourly candles)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(curve)
    drawdowns = (peak - curve) / peak
    max_dd = float(np.max(drawdowns)) * 100 if len(drawdowns) > 0 else 0

    # Trade statistics
    pnls = [t.pnl for t in trades if t.side == "SELL"]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0
    avg_pnl = float(np.mean(pnls)) if pnls else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Sortino ratio
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 1 and np.std(neg_returns) > 0:
        sortino = (np.mean(returns) / np.std(neg_returns)) * np.sqrt(365 * 24)
    else:
        sortino = 0.0

    # Calmar ratio
    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    return {
        "initial_capital": result.initial_capital,
        "final_value": final,
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd,
        "total_trades": len(pnls),
        "win_rate_pct": win_rate,
        "avg_trade_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0,
    }


def _empty_metrics(initial: float) -> dict:
    return {
        "initial_capital": initial,
        "final_value": initial,
        "total_return_pct": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "max_drawdown_pct": 0,
        "total_trades": 0,
        "win_rate_pct": 0,
        "avg_trade_pnl": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 0,
        "best_trade": 0,
        "worst_trade": 0,
    }
