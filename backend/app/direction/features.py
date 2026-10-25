"""Features for the direction models, from 30-minute candles plus 4-hour and
daily candles built from them.

A decision is made when a 30-minute candle CLOSES. Every feature for that
decision uses only candles that had closed by then: 4-hour and daily values
come from the last 4-hour / daily candle that had fully closed (UTC days).

Two inputs per decision:
  * TABULAR features (FEATURES) for the gradient-boosted trees, also given to
    the neural network as context;
  * a SEQUENCE of the last SEQ_LEN 30-minute candles (SEQ_COLS per candle) for
    the LSTM, which reads them in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.candle_store import resample_candles
from app.strategy.indicators import atr, ema
from app.strategy.orb_features import _AsOf

NY = "America/New_York"
BAR = pd.Timedelta(minutes=30)
NY_OPEN_MIN, NY_LAST_BAR_MIN = 9 * 60 + 30, 15 * 60 + 30   # decisions on candles opening 9:30..15:30 ET
SEQ_LEN = 48                                              # 24 hours of 30-minute candles
SEQ_COLS = ["s_ret", "s_range", "s_clv", "s_volz"]

FEATURES = [
    # 30-minute
    "r30_1", "r30_2", "r30_4", "r30_8", "r30_16", "r30_48", "atr30_pct", "pos48", "vol_ratio48",
    "rsi30", "ema20_30", "ema50_30",
    # New York session so far
    "ret_since_ny_open", "ny_range_pct", "ny_pos", "minutes_since_ny_open", "weekday",
    # 4-hour (last closed)
    "r4h_1", "r4h_3", "r4h_6", "ema20_4h", "ema50_4h", "atr4h_pct", "rsi4h",
    # daily (last closed UTC day)
    "r1d_1", "r1d_3", "r1d_7", "r1d_20", "ema20_1d", "ema50_1d", "atr1d_pct", "rsi1d",
    "dist_prev_day_high", "dist_prev_day_low",
    # the other coin (BTC for ETH, ETH for BTC)
    "x_r_2h", "x_r_24h", "x_r_4d",
    # funding and identity
    "funding_bps", "is_eth",
]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _pct(a, b):
    return (np.asarray(a, float) / np.asarray(b, float) - 1) * 100


def _higher_tf(df: pd.DataFrame, interval: str, known_after: pd.Timedelta) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "contracts", "quote_volume"]
    h = resample_candles(df[cols], interval)
    c = h["close"]
    lc = np.log(c)
    out = pd.DataFrame({"known_at": h["timestamp"] + known_after, "close": c, "high": h["high"],
                        "low": h["low"]})
    for k in (1, 3, 6, 7, 20):
        out[f"r_{k}"] = (lc - lc.shift(k)) * 100
    out["ema20"] = (c / ema(c, 20) - 1) * 100
    out["ema50"] = (c / ema(c, 50) - 1) * 100
    out["atr_pct"] = atr(h, 14) / c * 100
    out["rsi"] = rsi(c, 14)
    return out


def build_features(df: pd.DataFrame, symbol: str, other: pd.DataFrame | None = None,
                   funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per 30-minute candle (same index as df). Columns: FEATURES, SEQ_COLS,
    plus 'decision_time' (candle close, UTC), 'ny_date', 'is_decision' (a NY-session candle)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    lc = np.log(c)
    t_close = df["timestamp"] + BAR
    f = pd.DataFrame(index=df.index)
    f["decision_time"] = t_close

    for k in (1, 2, 4, 8, 16, 48):
        f[f"r30_{k}"] = (lc - lc.shift(k)) * 100
    f["atr30_pct"] = atr(df, 14) / c * 100
    hi48, lo48 = h.rolling(48).max(), l.rolling(48).min()
    f["pos48"] = (c - lo48) / (hi48 - lo48)
    f["vol_ratio48"] = v / v.shift(1).rolling(48, min_periods=24).mean()
    f["rsi30"] = rsi(c, 14)
    f["ema20_30"] = (c / ema(c, 20) - 1) * 100
    f["ema50_30"] = (c / ema(c, 50) - 1) * 100

    # New York session so far (from the 9:30 ET candle up to this candle)
    ny = df["timestamp"].dt.tz_convert(NY)
    ny_min = ny.dt.hour * 60 + ny.dt.minute
    ny_date = ny.dt.date.astype(str)
    in_session = (ny_min >= NY_OPEN_MIN) & (ny_min < 16 * 60)
    grp = ny_date.where(in_session)
    s_open = df["open"].where(in_session & (ny_min == NY_OPEN_MIN)).groupby(grp).transform("first")
    s_hi = h.where(in_session).groupby(grp).cummax()
    s_lo = l.where(in_session).groupby(grp).cummin()
    f["ret_since_ny_open"] = _pct(c, s_open)
    f["ny_range_pct"] = (s_hi - s_lo) / c * 100
    f["ny_pos"] = (c - s_lo) / (s_hi - s_lo)
    f["minutes_since_ny_open"] = (ny_min + 30 - NY_OPEN_MIN).where(in_session)
    f["weekday"] = ny.dt.weekday
    f["ny_date"] = ny_date
    f["is_decision"] = in_session & (ny_min <= NY_LAST_BAR_MIN)

    # 4-hour and daily, last closed candle only
    for tf, known_after, prefix, ks in (("4h", pd.Timedelta(hours=4), "4h", (1, 3, 6)),
                                        ("1d", pd.Timedelta(days=1), "1d", (1, 3, 7, 20))):
        hx = _higher_tf(df, tf, known_after)
        at = lambda col: _AsOf(hx["known_at"], hx[col])(t_close)  # noqa: E731
        for k in ks:
            f[f"r{prefix}_{k}"] = at(f"r_{k}")
        f[f"ema20_{prefix}"] = at("ema20")
        f[f"ema50_{prefix}"] = at("ema50")
        f[f"atr{prefix}_pct"] = at("atr_pct")
        f[f"rsi{prefix}"] = at("rsi")
        if tf == "1d":
            f["dist_prev_day_high"] = _pct(c, at("high"))
            f["dist_prev_day_low"] = _pct(c, at("low"))

    # The other coin, using only its closed candles
    if other is not None and len(other):
        o = other.sort_values("timestamp")
        px = _AsOf(o["timestamp"] + BAR, o["close"])
        now = px(t_close)
        f["x_r_2h"] = _pct(now, px(t_close - pd.Timedelta(hours=2)))
        f["x_r_24h"] = _pct(now, px(t_close - pd.Timedelta(hours=24)))
        f["x_r_4d"] = _pct(now, px(t_close - pd.Timedelta(days=4)))
    else:
        f["x_r_2h"] = f["x_r_24h"] = f["x_r_4d"] = np.nan

    if funding is not None and len(funding):
        f["funding_bps"] = _AsOf(funding["time"], funding["rate"])(t_close) * 1e4
    else:
        f["funding_bps"] = np.nan
    f["is_eth"] = float(symbol.upper().startswith("ETH"))

    # Per-candle sequence inputs for the LSTM (scale-free)
    lv = np.log(v.clip(lower=1e-12))
    f["s_ret"] = ((lc - lc.shift(1)) * 100).clip(-10, 10)
    f["s_range"] = ((h - l) / c * 100).clip(0, 20)
    f["s_clv"] = ((c - l) / (h - l).replace(0, np.nan) - 0.5).fillna(0)
    f["s_volz"] = ((lv - lv.rolling(48, min_periods=12).mean())
                   / lv.rolling(48, min_periods=12).std()).clip(-5, 5)

    return f.replace([np.inf, -np.inf], np.nan)


def sequences(f: pd.DataFrame, rows: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Array [len(rows), seq_len, len(SEQ_COLS)] of the candles ending at each row
    (the decision candle is the last). Missing values become 0."""
    arr = f[SEQ_COLS].to_numpy(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    idx = np.asarray(rows)[:, None] + np.arange(-seq_len + 1, 1)[None, :]
    idx = np.clip(idx, 0, len(arr) - 1)
    out = arr[idx]
    out[np.asarray(rows) < seq_len - 1] = 0.0     # not enough history yet
    return out
