"""Setup rules. Each returns, per bar, whether a long or short setup appeared
at that bar's close, how to enter, and where the stop goes.

Pullback in a trend (limit entry):
  Higher timeframe trending; on the base timeframe EMA20 is on the trend side
  of EMA50 and price closes on the trend side of EMA20, no more than 1 ATR
  away. Place a limit order at EMA20 (the pullback level), valid for a few bars.
  Stop: beyond the 10-bar swing low/high, at least 0.75 ATR and at most 3 ATR
  from entry.

Breakout on volume (market entry next bar):
  Higher timeframe trending; the bar closes beyond the previous 20-bar
  high/low on at least 1.5x average volume. Enter at the next bar's open.
  Stop: 0.5 ATR back inside the broken level (0.75-3 ATR from the close).

Nothing here looks at future bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.indicators import atr, ema, higher_tf_trend

ATR_N, FAST, SLOW = 14, 20, 50
MIN_STOP_ATR, MAX_STOP_ATR = 0.75, 3.0


def features(df: pd.DataFrame, interval: str, trend_interval: str) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["atr"] = atr(df, ATR_N)
    f["ema_fast"] = ema(df["close"], FAST)
    f["ema_slow"] = ema(df["close"], SLOW)
    f["swing_low"] = df["low"].rolling(10).min()
    f["swing_high"] = df["high"].rolling(10).max()
    f["donch_high"] = df["high"].shift(1).rolling(20).max()
    f["donch_low"] = df["low"].shift(1).rolling(20).min()
    f["vol_ratio"] = df["volume"] / df["volume"].shift(1).rolling(20).mean()
    tr = higher_tf_trend(df, interval, trend_interval)
    f["trend_up"], f["trend_down"] = tr["trend_up"], tr["trend_down"]
    return f


def _clip_stop(entry: pd.Series, raw_stop: pd.Series, a: pd.Series, side: int) -> pd.Series:
    dist = (entry - raw_stop) * side
    dist = dist.clip(lower=MIN_STOP_ATR * a, upper=MAX_STOP_ATR * a)
    return entry - side * dist


def pullback_signals(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    a, c = f["atr"], df["close"]
    long_ = (f["trend_up"] & (f["ema_fast"] > f["ema_slow"]) & (c > f["ema_fast"])
             & ((c - f["ema_fast"]) <= a))
    short = (f["trend_down"] & (f["ema_fast"] < f["ema_slow"]) & (c < f["ema_fast"])
             & ((f["ema_fast"] - c) <= a))
    entry = f["ema_fast"]
    out = pd.DataFrame(index=df.index)
    out["side"] = np.where(long_, 1, np.where(short, -1, 0))
    out["entry_type"] = "limit"
    out["entry"] = entry
    out["stop"] = np.where(long_, _clip_stop(entry, f["swing_low"] - 0.1 * a, a, 1),
                           _clip_stop(entry, f["swing_high"] + 0.1 * a, a, -1))
    out["atr"] = a
    out.loc[~(f["atr"].notna() & f["ema_slow"].notna()), "side"] = 0
    return out


def breakout_signals(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    a, c = f["atr"], df["close"]
    vol_ok = f["vol_ratio"] >= 1.5
    long_ = f["trend_up"] & (c > f["donch_high"]) & vol_ok & (c > f["ema_slow"])
    short = f["trend_down"] & (c < f["donch_low"]) & vol_ok & (c < f["ema_slow"])
    out = pd.DataFrame(index=df.index)
    out["side"] = np.where(long_, 1, np.where(short, -1, 0))
    out["entry_type"] = "market"
    out["entry"] = c   # reference only: actual fill is the next bar's open
    out["stop"] = np.where(long_, _clip_stop(c, f["donch_high"] - 0.5 * a, a, 1),
                           _clip_stop(c, f["donch_low"] + 0.5 * a, a, -1))
    out["atr"] = a
    out.loc[~(f["atr"].notna() & f["donch_high"].notna() & f["vol_ratio"].notna()), "side"] = 0
    return out


def signals(setup: str, df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    s = pullback_signals(df, f) if setup == "pullback" else breakout_signals(df, f)
    s["side"] = s["side"].astype(int)
    return s
