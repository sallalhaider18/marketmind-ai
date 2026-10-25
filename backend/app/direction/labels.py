"""What would have happened after each decision: a long and a short trade with
the same stop and target distance (1:1), entered at the next 30-minute candle's
open and held until the target, the stop, or 9:29 ET the next day.

Conservative, like the other backtests:
  * if a candle touches both the stop and the target, the stop counts;
  * a gap through the stop exits at the (worse) open;
  * entry and stop/time exits pay taker fee + slippage, targets pay maker fee;
  * funding is charged at each 8-hour settlement the trade spans.
Results are in R (1R = the stop distance), so BTC and ETH compare directly.

`up_first` (the model's learning target) = 1 if price reached +distance before
-distance, 0 if the reverse, NaN if neither or both inside one candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.engine import FUNDING_PERIOD, Costs

NY = "America/New_York"


def horizon_end(df: pd.DataFrame, rows: np.ndarray, ny_dates: pd.Series) -> np.ndarray:
    """Index of the last candle closing by 9:30 ET on the day after each decision's NY date."""
    ts = df["timestamp"].to_numpy("datetime64[ns]")
    ends = []
    cache = {}
    for d in ny_dates:
        if d not in cache:
            nd = pd.Timestamp(d) + pd.Timedelta(days=1)
            t = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day, hour=9, minute=30, tz=NY)
            # candles opening before 9:00 ET close by 9:30 ET (30-minute candles)
            limit = (t - pd.Timedelta(minutes=30)).tz_convert("UTC").tz_localize(None).to_datetime64()
            cache[d] = int(np.searchsorted(ts, limit, side="right")) - 1
        ends.append(cache[d])
    return np.asarray(ends)


def simulate(o, h, l, c, ts, start: int, end: int, side: int, dist: float, cost: Costs) -> tuple[float, str, int]:
    """One trade: market entry at o[start]; returns (R, exit reason, exit index)."""
    entry = o[start] * (1 + side * cost.slippage)
    stop, target = entry - side * dist, entry + side * dist
    seg = slice(start, end + 1)
    if side == 1:
        stop_hit = l[seg] <= stop
        tgt_hit = h[seg] >= target
    else:
        stop_hit = h[seg] >= stop
        tgt_hit = l[seg] <= target
    i_stop = int(np.argmax(stop_hit)) if stop_hit.any() else None
    i_tgt = int(np.argmax(tgt_hit)) if tgt_hit.any() else None
    if i_stop is not None and (i_tgt is None or i_stop <= i_tgt):
        j = start + i_stop
        gap = j > start and ((side == 1 and o[j] <= stop) or (side == -1 and o[j] >= stop))
        px = (o[j] if gap else stop) * (1 - side * cost.slippage)
        reason, fee = "stop", cost.taker_fee
    elif i_tgt is not None:
        j = start + i_tgt
        gap = j > start and ((side == 1 and o[j] >= target) or (side == -1 and o[j] <= target))
        px = o[j] if gap else target
        reason, fee = "target", cost.maker_fee
    else:
        j = end
        px = c[j] * (1 - side * cost.slippage)
        reason, fee = "next-day close (9:29 ET)", cost.taker_fee
    t_in, t_out = int(ts[start]), int(ts[j]) + 1800
    crossings = max(0, t_out // FUNDING_PERIOD - t_in // FUNDING_PERIOD)
    pnl = side * (px - entry) - entry * cost.taker_fee - px * fee - crossings * cost.funding_per_8h * entry
    return pnl / dist, reason, j


def label_decisions(df: pd.DataFrame, rows: np.ndarray, ny_dates: pd.Series, dist: float,
                    cost: Costs = Costs()) -> pd.DataFrame:
    """For each decision candle index in `rows`: outcomes of a long and a short trade."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = ((df["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).to_numpy("int64")
    ends = horizon_end(df, rows, ny_dates)
    out = []
    for i, end in zip(rows, ends):
        start = i + 1
        if start >= len(df) or end < start:
            out.append((np.nan, np.nan, None, None, np.nan, -1, np.nan))
            continue
        lr, lreason, lj = simulate(o, h, l, c, ts, start, end, 1, dist, cost)
        sr, sreason, sj = simulate(o, h, l, c, ts, start, end, -1, dist, cost)
        # learning target from the raw path (no costs, from the entry candle's open)
        ref = o[start]
        up = h[start:end + 1] >= ref + dist
        dn = l[start:end + 1] <= ref - dist
        iu = int(np.argmax(up)) if up.any() else None
        idn = int(np.argmax(dn)) if dn.any() else None
        if iu is None and idn is None:
            y = np.nan
        elif idn is None or (iu is not None and iu < idn):
            y = 1.0
        elif iu is None or idn < iu:
            y = 0.0
        else:
            y = np.nan   # both in the same candle: order unknown
        out.append((lr, sr, lreason, sreason, y, end, o[start]))
    lab = pd.DataFrame(out, columns=["long_r", "short_r", "long_exit", "short_exit", "up_first",
                                     "horizon_end", "entry_ref"])
    lab.index = rows
    return lab
