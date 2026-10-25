"""Performance metrics. Every figure is after fees, slippage and funding."""
from __future__ import annotations

import math

import pandas as pd


def summarize(trades: pd.DataFrame, start_equity: float = 10_000.0) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None,
                "avg_win_r": None, "avg_loss_r": None, "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "avg_bars_held": None, "long_trades": 0, "short_trades": 0, "costs_share_of_risk": None,
                "t_stat": None}
    r = trades["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = -trades.loc[trades["pnl_usd"] <= 0, "pnl_usd"].sum()
    # Equity curve from per-trade P&L (start of this slice = start_equity)
    eq = start_equity + trades["pnl_usd"].cumsum()
    peak = pd.concat([pd.Series([start_equity]), eq]).cummax().iloc[1:].to_numpy()
    dd = ((eq.to_numpy() - peak) / peak).min()
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "expectancy_r": r.mean(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else math.inf,
        "avg_win_r": wins.mean() if len(wins) else None,
        "avg_loss_r": losses.mean() if len(losses) else None,
        "total_return_pct": trades["pnl_usd"].sum() / start_equity * 100,
        "max_drawdown_pct": min(0.0, dd) * 100,
        "avg_bars_held": trades["bars_held"].mean(),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "costs_share_of_risk": (trades["costs_usd"] / trades["risk_usd"]).mean(),
        # How many standard errors the average result is above zero. Around 2 or more
        # means it is unlikely to be luck; near 0 means it could easily be noise.
        "t_stat": (r.mean() / (r.std(ddof=1) / math.sqrt(n))) if n > 1 and r.std(ddof=1) > 0 else None,
    }


def by_group(trades: pd.DataFrame, col: str) -> dict:
    out = {}
    for k, g in trades.groupby(col):
        out[str(k)] = {"trades": len(g), "win_rate": (g["r"] > 0).mean(), "expectancy_r": g["r"].mean()}
    return out
