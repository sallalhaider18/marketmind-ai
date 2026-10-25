"""New York session opening-range breakout (ORB), day-trade only.

Rules (times in New York time, America/New_York, so daylight saving is handled):
  * Weekdays only (crypto trades every day, but the New York open happens on weekdays).
  * Opening range (OR) = high and low of 9:30-9:45 ET (the first 15 minutes of
    the US cash session). Range = OR high - OR low.
  * Breakout = the first candle after 9:45 that CLOSES above the OR high (long)
    or below the OR low (short), and opens before 11:00 ET. One trade per day.
  * Entry, two variants:
      - "close":  market order at the next candle's open.
      - "retest": limit order at the broken range edge, valid until 11:00 ET.
                  Cancelled if price reaches the first target without retesting.
  * Stop, two variants:
      - "range": the opposite side of the opening range.
      - "mid":   the middle of the opening range (a tighter, objective stand-in
                 for "past the recent micro-swing").
  * Target, six variants:
      measured from the broken edge, in opening ranges
      - "1x":      edge + 1 x range.
      - "2x":      edge + 2 x range.
      - "partial": half closed at 1 x range, stop moved to entry, rest at 2 x range.
      measured from the entry, in stop distances (reward-to-risk)
      - "1:1":     target as far from entry as the stop (win and loss the same size).
      - "1.5:1":   target 1.5 x the stop distance.
      - "2:1":     target 2 x the stop distance.
  * Fixed-distance mode (OrbConfig.fixed_distance or fixed_pct): the stop is that
    distance from the entry and the target fixed_rr times as far, e.g. BTC stop
    500 / target 1,000 with fixed_distance=500, fixed_rr=2 (default 1:1).
    Overrides the stop and target variants above.
  * Exits when neither stop nor target is hit:
      - default: closed at the 15:55-16:00 ET candle's close (day trade);
      - OrbConfig.exit_next_open: held until 9:29 ET on the next trading day, i.e.
        the close of the last candle before the next 9:30 ET open. With
        OrbConfig.weekends the next trading day is simply tomorrow (Saturday and
        Sunday are traded too); without it, Friday trades are held to Monday;
      - OrbConfig.hold_hours: held up to that many hours.
    No new trade is taken while one is still open.

Fills are conservative, as in the main backtester: stop assumed first when a
candle touches both stop and target; a limit entry cannot take profit on its
fill candle; gaps through the stop exit at the worse open; after a partial
take-profit, a candle that also returns to the entry counts as stopped at
break-even.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from app.backtest.engine import FUNDING_PERIOD, Costs

NY = "America/New_York"
OR_START, OR_END = time(9, 30), time(9, 45)
ENTRY_CUTOFF, SESSION_END = time(11, 0), time(16, 0)

ENTRIES = ("close", "retest")
STOPS = ("range", "mid")
TARGETS = ("1x", "2x", "partial", "1:1", "1.5:1", "2:1")
# "a:1" targets = reward-to-risk: the target is `a` times the actual stop distance
RR_TARGETS = {"1:1": 1.0, "1.5:1": 1.5, "2:1": 2.0}


@dataclass(frozen=True)
class OrbConfig:
    entry: str = "retest"
    stop: str = "range"
    target: str = "partial"
    risk_pct: float = 0.01
    start_equity: float = 10_000.0
    max_leverage: float = 10.0
    costs: Costs = Costs()
    fixed_distance: float | None = None   # price units, e.g. 1000 for BTC: stop and target this far away
    fixed_pct: float | None = None        # or as a fraction of entry, e.g. 0.012 = 1.2%
    fixed_rr: float = 1.0                 # target = fixed_rr x stop distance (2.0: stop 500, target 1000)
    hold_hours: float | None = None       # None = close at 16:00 ET; else hold up to this many hours
    exit_next_open: bool = False          # hold until 9:29 ET on the next trading day (overrides hold_hours)
    weekends: bool = False                # also trade Saturday and Sunday (then "next trading day" = tomorrow)

    @property
    def fixed(self) -> bool:
        return self.fixed_distance is not None or self.fixed_pct is not None


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def session_days(df: pd.DataFrame, bar_minutes: int, weekends: bool = False):
    """Yield (ny_date, day_bars) for weekdays (or every day) with a complete opening range.
    day_bars: candles from 9:30 to 16:00 ET with a column `ny_min` (minutes since midnight ET)."""
    ny = df["timestamp"].dt.tz_convert(NY)
    d = df.assign(ny_date=ny.dt.date, ny_min=ny.dt.hour * 60 + ny.dt.minute, weekday=ny.dt.weekday,
                  _pos=np.arange(len(df)))
    days_ok = d["weekday"] < (7 if weekends else 5)
    d = d[days_ok & (d["ny_min"] >= _minutes(OR_START)) & (d["ny_min"] < _minutes(SESSION_END))]
    need = (_minutes(OR_END) - _minutes(OR_START)) // bar_minutes
    for day, g in d.groupby("ny_date", sort=True):
        or_bars = g[g["ny_min"] < _minutes(OR_END)]
        if len(or_bars) != need or or_bars["ny_min"].iloc[0] != _minutes(OR_START):
            continue  # missing data around the open: skip the day rather than guess
        yield day, g.reset_index(drop=True)


def run_orb(df: pd.DataFrame, bar_minutes: int, cfg: OrbConfig) -> tuple[list[dict], dict]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    FO, FH, FL, FC = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    FTS = ((df["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).to_numpy("int64")
    hold_bars = None if cfg.hold_hours is None else max(1, int(cfg.hold_hours * 60 / bar_minutes))
    busy_until = -1   # with hold_hours, no new trade while one is open
    cost = cfg.costs
    equity = cfg.start_equity
    trades: list[dict] = []
    stats = {"days": 0, "breakouts": 0, "retest_missed": 0, "retest_expired": 0, "skipped": 0,
             "skipped_open_trade": 0}
    ranges: list[float] = []   # relative range sizes, for "narrow / normal / wide day" labels

    for day, g in session_days(df, bar_minutes, cfg.weekends):
        stats["days"] += 1
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        mins = g["ny_min"].to_numpy()
        ts = ((g["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).to_numpy("int64")
        is_or = mins < _minutes(OR_END)
        orh, orl = h[is_or].max(), l[is_or].min()
        rng = orh - orl
        if rng <= 0:
            continue
        rel = rng / ((orh + orl) / 2)
        hist = np.array(ranges[-60:])
        ranges.append(rel)
        if len(hist) >= 20:
            q = (hist < rel).mean()
            range_regime = "narrow range day" if q < 0.25 else "wide range day" if q > 0.75 else "normal range day"
        else:
            range_regime = "normal range day"

        # 1) breakout: first close outside the range, candle opening before 11:00
        first = int(is_or.sum())
        b = None
        for i in range(first, len(g)):
            if mins[i] >= _minutes(ENTRY_CUTOFF):
                break
            if c[i] > orh or c[i] < orl:
                b = i
                break
        if b is None or b + 1 >= len(g):
            continue
        stats["breakouts"] += 1
        side = 1 if c[b] > orh else -1
        edge = orh if side == 1 else orl
        stop = (orl if side == 1 else orh) if cfg.stop == "range" else (orh + orl) / 2
        tp1 = edge + side * rng
        tp2 = edge + side * 2 * rng
        final_tp = tp1 if cfg.target == "1x" else tp2

        # 2) entry
        fill_i = entry = None
        if cfg.entry == "close":
            fill_i = b + 1
            entry = o[fill_i] * (1 + side * cost.slippage)
            entry_fee_rate = cost.taker_fee
        else:
            entry_fee_rate = cost.maker_fee
            for i in range(b + 1, len(g)):
                if mins[i] >= _minutes(ENTRY_CUTOFF):
                    stats["retest_expired"] += 1
                    break
                if (side == 1 and l[i] <= edge) or (side == -1 and h[i] >= edge):
                    fill_i = i
                    entry = min(o[i], edge) if side == 1 else max(o[i], edge)
                    break
                if (side == 1 and h[i] >= tp1) or (side == -1 and l[i] <= tp1):
                    stats["retest_missed"] += 1
                    break
        if fill_i is None:
            continue
        if ts[fill_i] < busy_until:
            stats["skipped_open_trade"] += 1
            continue
        if cfg.fixed:
            dist = cfg.fixed_distance if cfg.fixed_distance is not None else entry * cfg.fixed_pct
            stop = entry - side * dist
            final_tp = entry + side * dist * cfg.fixed_rr
        risk_unit = (entry - stop) * side
        if cfg.target in RR_TARGETS and not cfg.fixed:
            final_tp = entry + side * RR_TARGETS[cfg.target] * risk_unit
        partial_mode = cfg.target == "partial" and not cfg.fixed
        first_target = tp1 if partial_mode else final_tp
        if risk_unit <= 0 or (first_target - entry) * side <= 0:
            stats["skipped"] += 1   # price already past the stop or the target at entry
            continue
        risk_usd = equity * cfg.risk_pct
        qty = risk_usd / risk_unit
        if qty * entry > equity * cfg.max_leverage:
            qty = equity * cfg.max_leverage / entry
            risk_usd = qty * risk_unit

        # 3) manage the trade, over today's session or (hold_hours) over the following candles
        p0 = int(g["_pos"].iloc[fill_i])
        if cfg.exit_next_open:
            nxt = pd.Timestamp(day) + pd.Timedelta(days=1)
            while not cfg.weekends and nxt.weekday() >= 5:   # weekdays only: Friday -> Monday
                nxt += pd.Timedelta(days=1)
            next_open = pd.Timestamp(year=nxt.year, month=nxt.month, day=nxt.day, hour=9, minute=30, tz=NY)
            p1 = int(np.searchsorted(FTS, int(next_open.tz_convert("UTC").timestamp()), side="left")) - 1
            p1 = max(p0, min(p1, len(df) - 1))
        elif hold_bars is None:
            p1 = int(g["_pos"].iloc[-1])
        else:
            p1 = min(len(df) - 1, p0 + hold_bars - 1)
        o, h, l, c, ts = FO[p0:p1 + 1], FH[p0:p1 + 1], FL[p0:p1 + 1], FC[p0:p1 + 1], FTS[p0:p1 + 1]
        fill_ts = int(ts[0])
        fill_i = 0
        legs: list[tuple[float, float, bool]] = []   # (qty, exit price, taker?)
        remaining = qty
        cur_stop = stop
        partial_done = False
        reason = None
        exit_i = fill_i
        for i in range(fill_i, len(o)):
            exit_i = i
            on_fill_bar = i == fill_i
            # stop (or gap through it)
            gap = (side == 1 and o[i] <= cur_stop) or (side == -1 and o[i] >= cur_stop)
            hit = (side == 1 and l[i] <= cur_stop) or (side == -1 and h[i] >= cur_stop)
            if not on_fill_bar and gap:
                legs.append((remaining, o[i] * (1 - side * cost.slippage), True))
                reason = "breakeven" if partial_done else "stop"
                break
            if hit:
                legs.append((remaining, cur_stop * (1 - side * cost.slippage), True))
                reason = "breakeven" if partial_done else "stop"
                break
            # targets (not on a limit order's fill candle)
            if not (on_fill_bar and cfg.entry == "retest"):
                if partial_mode and not partial_done and \
                        ((side == 1 and h[i] >= tp1) or (side == -1 and l[i] <= tp1)):
                    half = qty / 2
                    legs.append((half, tp1, False))
                    remaining -= half
                    partial_done = True
                    cur_stop = entry
                    if (side == 1 and h[i] >= tp2) or (side == -1 and l[i] <= tp2):
                        legs.append((remaining, tp2, False))
                        reason = "target"
                        break
                    if (side == 1 and l[i] <= entry) or (side == -1 and h[i] >= entry):
                        legs.append((remaining, entry * (1 - side * cost.slippage), True))
                        reason = "breakeven"   # conservative: back to entry on the same candle
                        break
                    continue
                tp = tp2 if partial_mode else final_tp
                if (side == 1 and h[i] >= tp) or (side == -1 and l[i] <= tp):
                    legs.append((remaining, tp, False))
                    reason = "target"
                    break
        if reason is None:   # 16:00 ET (or the hold limit): close what is left at the last close
            legs.append((remaining, c[exit_i] * (1 - side * cost.slippage), True))
            limit = ("next-day close (9:29 ET)" if cfg.exit_next_open
                     else "session close" if hold_bars is None else "time limit")
            reason = f"partial + {limit}" if partial_done else limit

        gross = sum(side * (px - entry) * q for q, px, _ in legs)
        fees = entry * qty * entry_fee_rate + sum(px * q * (cost.taker_fee if tk else cost.maker_fee)
                                                  for q, px, tk in legs)
        t_in, t_out = fill_ts, int(ts[exit_i]) + bar_minutes * 60
        busy_until = t_out
        funding = max(0, t_out // FUNDING_PERIOD - t_in // FUNDING_PERIOD) * cost.funding_per_8h * entry * qty
        pnl = gross - fees - funding
        equity += pnl
        trades.append({
            "entry_time": pd.Timestamp(t_in, unit="s", tz="UTC"),
            "exit_time": pd.Timestamp(t_out, unit="s", tz="UTC"),
            "ny_date": str(day), "side": "long" if side == 1 else "short",
            "entry_type": "market" if cfg.entry == "close" else "limit",
            "entry": entry, "stop": stop, "target": final_tp, "or_high": orh, "or_low": orl,
            # The breakout candle: everything a filter may know is known when it closes
            "breakout_time": pd.Timestamp(int(FTS[int(g["_pos"].iloc[b])]), unit="s", tz="UTC"),
            "breakout_close": float(g["close"].iloc[b]),
            "exit_reason": reason, "bars_held": exit_i - fill_i + 1, "qty": qty, "risk_usd": risk_usd,
            "costs_usd": fees + funding, "pnl_usd": pnl, "r": pnl / risk_usd, "equity_after": equity,
            "range_regime": range_regime, "weekday": pd.Timestamp(day).day_name(),
        })
    stats["final_equity"] = equity
    return trades, stats
