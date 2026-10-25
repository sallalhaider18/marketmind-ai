"""Indicators. Every value at row t uses only data up to and including bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.adapters.base import INTERVAL_SECONDS


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()   # Wilder


def higher_tf_trend(base: pd.DataFrame, base_interval: str, trend_interval: str,
                    fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Trend on a higher timeframe, aligned to base bars WITHOUT look-ahead:
    a higher-timeframe bar is only used once it has closed."""
    from app.data.candle_store import resample_candles  # local import avoids a cycle

    t = resample_candles(base[["timestamp", "open", "high", "low", "close", "volume", "contracts",
                               "quote_volume"]], trend_interval)
    t = t.assign(t_ema_fast=ema(t["close"], fast), t_ema_slow=ema(t["close"], slow), t_close=t["close"])
    # The higher-timeframe bar is known when it closes
    t["available_at"] = t["timestamp"] + pd.Timedelta(seconds=INTERVAL_SECONDS[trend_interval])
    # A base bar's signal is evaluated when the base bar closes
    b = pd.DataFrame({"close_time": base["timestamp"] + pd.Timedelta(seconds=INTERVAL_SECONDS[base_interval])})
    b["_row"] = np.arange(len(b))
    merged = pd.merge_asof(b.sort_values("close_time"),
                           t[["available_at", "t_close", "t_ema_fast", "t_ema_slow"]].sort_values("available_at"),
                           left_on="close_time", right_on="available_at", direction="backward")
    merged = merged.sort_values("_row").reset_index(drop=True)
    up = (merged["t_close"] > merged["t_ema_slow"]) & (merged["t_ema_fast"] > merged["t_ema_slow"])
    down = (merged["t_close"] < merged["t_ema_slow"]) & (merged["t_ema_fast"] < merged["t_ema_slow"])
    return pd.DataFrame({"trend_up": up.to_numpy(), "trend_down": down.to_numpy()}, index=base.index)
