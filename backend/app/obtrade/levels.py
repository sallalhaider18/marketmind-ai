"""Strong support and resistance levels from the DAILY and 4-HOUR charts.

1. Turning points: swing highs / lows on daily candles (2 candles each side,
   last ~180 days) and on 4-hour candles (5 candles each side, last ~90 days).
   A daily turning point counts 3, a 4-hour one counts 1 ("score").
2. Turning points within CLUSTER_PCT of each other are merged into one level.
3. Only the strongest levels are kept: highest score first, each at least
   MIN_GAP_PCT away from a stronger one, at most MAX_EACH_SIDE above and below
   the current price, and a score of at least MIN_SCORE (one daily turn, or
   two 4-hour turns).
Only fully closed candles are used, and a turning point only counts once the
candles after it have closed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

K_4H, K_1D = 5, 2
LOOKBACK_4H, LOOKBACK_1D = 540, 180
WEIGHT_4H, WEIGHT_1D = 1, 3
CLUSTER_PCT = 0.004       # merge turning points within 0.4%
MIN_GAP_PCT = 0.008       # kept levels at least 0.8% apart
MAX_EACH_SIDE = 3
MIN_SCORE = 2


@dataclass
class Level:
    price: float
    touches: int          # turning points merged into this level
    last_time: str        # most recent turning point (ISO)
    score: int = 0        # daily turns x3 + 4h turns
    daily: int = 0        # how many of the touches are daily turning points

    def as_dict(self, current: float | None = None) -> dict:
        d = {"price": self.price, "touches": self.touches, "score": self.score, "daily": self.daily,
             "last_time": self.last_time}
        if current is not None:
            d["kind"] = "resistance" if self.price > current else "support"
        return d


def _pivots(df: pd.DataFrame, k: int, lookback: int) -> list[tuple[float, pd.Timestamp]]:
    df = df.sort_values("timestamp").tail(lookback).reset_index(drop=True)
    if len(df) < 2 * k + 1:
        return []
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    win = 2 * k + 1
    hi_max = pd.Series(h).rolling(win, center=True).max().to_numpy()
    lo_min = pd.Series(l).rolling(win, center=True).min().to_numpy()
    out = []
    for i in range(k, len(df) - k):
        if h[i] == hi_max[i]:
            out.append((h[i], df["timestamp"].iloc[i]))
        if l[i] == lo_min[i]:
            out.append((l[i], df["timestamp"].iloc[i]))
    return out


def find_levels(df4h: pd.DataFrame | None, df1d: pd.DataFrame | None = None, current: float | None = None,
                max_each_side: int = MAX_EACH_SIDE, min_gap_pct: float = MIN_GAP_PCT,
                min_score: int = MIN_SCORE) -> list[Level]:
    pts: list[tuple[float, pd.Timestamp, int]] = []
    if df4h is not None and len(df4h):
        pts += [(p, t, WEIGHT_4H) for p, t in _pivots(df4h, K_4H, LOOKBACK_4H)]
        current = current if current is not None else float(df4h.sort_values("timestamp")["close"].iloc[-1])
    if df1d is not None and len(df1d):
        pts += [(p, t, WEIGHT_1D) for p, t in _pivots(df1d, K_1D, LOOKBACK_1D)]
        if current is None:
            current = float(df1d.sort_values("timestamp")["close"].iloc[-1])
    if not pts:
        return []
    pts.sort(key=lambda x: x[0])
    clusters: list[list[tuple[float, pd.Timestamp, int]]] = [[pts[0]]]
    for p in pts[1:]:
        ref = np.average([q[0] for q in clusters[-1]], weights=[q[2] for q in clusters[-1]])
        if abs(p[0] - ref) / ref <= CLUSTER_PCT:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [Level(price=float(np.average([q[0] for q in c], weights=[q[2] for q in c])), touches=len(c),
                    last_time=max(q[1] for q in c).isoformat(), score=int(sum(q[2] for q in c)),
                    daily=sum(1 for q in c if q[2] == WEIGHT_1D))
              for c in clusters]
    levels = [lv for lv in levels if lv.score >= min_score]
    # strongest first; recent turning points break ties
    levels.sort(key=lambda lv: (lv.score, lv.last_time), reverse=True)
    kept: list[Level] = []
    for lv in levels:
        if all(abs(lv.price - k.price) / k.price >= min_gap_pct for k in kept):
            kept.append(lv)
    below = [lv for lv in kept if lv.price < current][:max_each_side]
    above = [lv for lv in kept if lv.price > current][:max_each_side]
    return sorted(below + above, key=lambda lv: lv.price)


def split_levels(levels: list[Level], price: float) -> tuple[list[Level], list[Level]]:
    """(supports below price, nearest first), (resistances above price, nearest first)."""
    sup = sorted([lv for lv in levels if lv.price < price], key=lambda x: -x.price)
    res = sorted([lv for lv in levels if lv.price > price], key=lambda x: x.price)
    return sup, res
