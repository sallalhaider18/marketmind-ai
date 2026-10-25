"""Features describing each ORB breakout, using only information available
when the breakout candle CLOSES ("known time"). Directional features are
multiplied by the trade side, so a positive value always means "in favour
of the trade" (e.g. trend_ema50_1h_s > 0 = trading with the 1h trend).

Feature list (FEATURES):
  or_rel_pct          opening-range size as % of price
  or_vs_20d           opening range / average of the previous 20 opening ranges
  or_vs_atr1h         opening range / 1-hour ATR(14)
  or_vol_vs_20d       volume during the opening range / previous 20 days' average
  bo_vol_ratio        breakout candle volume / average candle volume over the previous 24 h
  bo_strength         how far the breakout candle closed beyond the range edge, in ranges
  bo_minutes          minutes after 9:45 ET that the breakout candle opened
  asia_ret_s          19:00 (prev day) -> 03:00 ET move, %, signed
  london_ret_s        03:00 -> 08:00 ET move, %, signed
  preopen_ret_s       08:00 -> 09:30 ET move, %, signed
  overnight_ret_s     16:00 (prev day) -> 09:30 ET move, %, signed
  dist_prev_extreme_s distance beyond yesterday's high (longs) / low (shorts), in ranges
  dist_prev_opposite_s distance from yesterday's low (longs) / high (shorts), in ranges
  trend_ema50_1h_s    distance from the 1-hour EMA50, %, signed
  prev_day_ret_s      return over the previous 24 h to 9:30 ET, %, signed
  ret_5d_s            return over the previous 5 days to 9:30 ET, %, signed
  funding_s           last settled funding rate in basis points, signed (positive = this side pays)
  weekday_num         0 = Monday ... 4 = Friday
  side_num            1 long, -1 short
  is_eth              1 for ETH, 0 for BTC
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from app.data.candle_store import resample_candles
from app.strategy.indicators import atr, ema
from app.strategy.orb import NY, session_days

FEATURES = [
    "or_rel_pct", "or_vs_20d", "or_vs_atr1h", "or_vol_vs_20d", "bo_vol_ratio", "bo_strength", "bo_minutes",
    "asia_ret_s", "london_ret_s", "preopen_ret_s", "overnight_ret_s", "dist_prev_extreme_s",
    "dist_prev_opposite_s", "trend_ema50_1h_s", "prev_day_ret_s", "ret_5d_s", "funding_s", "weekday_num",
    "side_num", "is_eth",
]


def _ns(ts) -> np.ndarray:
    return pd.DatetimeIndex(ts).as_unit("ns").asi8


class _AsOf:
    """Value of a series as of a time: the last value whose 'known at' time <= t."""

    def __init__(self, known_at, values):
        order = np.argsort(_ns(known_at))
        self.t = _ns(known_at)[order]
        self.v = np.asarray(values, dtype=float)[order]

    def __call__(self, times) -> np.ndarray:
        idx = np.searchsorted(self.t, _ns(times), side="right") - 1
        out = np.full(len(idx), np.nan)
        ok = idx >= 0
        out[ok] = self.v[idx[ok]]
        return out


def _ny(date, hh, mm, days=0) -> pd.Timestamp:
    d = pd.Timestamp(date) + pd.Timedelta(days=days)
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=NY).tz_convert("UTC")


def orb_features(df: pd.DataFrame, bar_minutes: int, trades: pd.DataFrame, symbol: str,
                 funding: pd.DataFrame | None = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=FEATURES)
    bar = pd.Timedelta(minutes=bar_minutes)
    df = df.sort_values("timestamp").reset_index(drop=True)
    close_time = df["timestamp"] + bar
    price_at = _AsOf(close_time, df["close"])                  # last close known at t

    # 1-hour EMA50 and ATR, known when each hour closes
    h1 = resample_candles(df[["timestamp", "open", "high", "low", "close", "volume", "contracts",
                              "quote_volume"]], "1h")
    h1_known = h1["timestamp"] + pd.Timedelta(hours=1)
    ema50_at = _AsOf(h1_known, ema(h1["close"], 50))
    atr1h_at = _AsOf(h1_known, atr(h1, 14))

    # Candle volume vs the previous 24 h
    n24 = int(24 * 60 / bar_minutes)
    vol_ratio = df["volume"] / df["volume"].shift(1).rolling(n24, min_periods=n24 // 2).mean()
    vol_ratio_by_time = pd.Series(vol_ratio.to_numpy(), index=df["timestamp"])

    # Opening ranges for every day (including days without a trade), for "vs previous 20 days"
    or_rows = []
    or_end_min = 9 * 60 + 45
    for day, g in session_days(df, bar_minutes):
        orb = g[g["ny_min"] < or_end_min]
        or_rows.append({"ny_date": str(day), "or_range": orb["high"].max() - orb["low"].min(),
                        "or_volume": orb["volume"].sum()})
    ors = pd.DataFrame(or_rows).set_index("ny_date")
    ors["or_range_20d"] = ors["or_range"].shift(1).rolling(20, min_periods=10).mean()
    ors["or_vol_20d"] = ors["or_volume"].shift(1).rolling(20, min_periods=10).mean()

    # Previous New York calendar day's high and low
    ny_date = df["timestamp"].dt.tz_convert(NY).dt.date.astype(str)
    daily = df.groupby(ny_date).agg(high=("high", "max"), low=("low", "min"))
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D")
    prev = daily.shift(1)

    fund_at = _AsOf(funding["time"], funding["rate"]) if funding is not None and len(funding) else None

    t = trades.reset_index(drop=True)
    s = np.where(t["side"] == "long", 1.0, -1.0)
    known = pd.DatetimeIndex(t["breakout_time"]) + bar
    rng_ = (t["or_high"] - t["or_low"]).to_numpy()
    mid = ((t["or_high"] + t["or_low"]) / 2).to_numpy()
    cb = t["breakout_close"].to_numpy()
    edge = np.where(s == 1, t["or_high"], t["or_low"])
    dates = t["ny_date"].tolist()

    def times(hh, mm, days=0):
        return pd.DatetimeIndex([_ny(d, hh, mm, days) for d in dates])

    p930, p800, p300 = price_at(times(9, 30)), price_at(times(8, 0)), price_at(times(3, 0))
    p_prev1900, p_prev1600 = price_at(times(19, 0, -1)), price_at(times(16, 0, -1))
    p_prev930, p_5d930 = price_at(times(9, 30, -1)), price_at(times(9, 30, -5))

    def pct(a, b):
        return (a / b - 1) * 100

    or_info = ors.reindex(dates)
    pv = prev.reindex(pd.to_datetime(dates))
    prev_high, prev_low = pv["high"].to_numpy(), pv["low"].to_numpy()

    f = pd.DataFrame({
        "or_rel_pct": rng_ / mid * 100,
        "or_vs_20d": rng_ / or_info["or_range_20d"].to_numpy(),
        "or_vs_atr1h": rng_ / atr1h_at(known),
        "or_vol_vs_20d": or_info["or_volume"].to_numpy() / or_info["or_vol_20d"].to_numpy(),
        "bo_vol_ratio": vol_ratio_by_time.reindex(pd.DatetimeIndex(t["breakout_time"])).to_numpy(),
        "bo_strength": s * (cb - edge) / rng_,
        "bo_minutes": ((pd.DatetimeIndex(t["breakout_time"]).tz_convert(NY).hour * 60
                        + pd.DatetimeIndex(t["breakout_time"]).tz_convert(NY).minute) - or_end_min).to_numpy(),
        "asia_ret_s": s * pct(p300, p_prev1900),
        "london_ret_s": s * pct(p800, p300),
        "preopen_ret_s": s * pct(p930, p800),
        "overnight_ret_s": s * pct(p930, p_prev1600),
        "dist_prev_extreme_s": np.where(s == 1, cb - prev_high, prev_low - cb) / rng_,
        "dist_prev_opposite_s": np.where(s == 1, cb - prev_low, prev_high - cb) / rng_,
        "trend_ema50_1h_s": s * pct(cb, ema50_at(known)),
        "prev_day_ret_s": s * pct(p930, p_prev930),
        "ret_5d_s": s * pct(p930, p_5d930),
        "funding_s": s * fund_at(known) * 1e4 if fund_at else np.full(len(t), np.nan),
        "weekday_num": pd.to_datetime(pd.Series(dates)).dt.weekday.to_numpy(),
        "side_num": s,
        "is_eth": float(symbol.upper().startswith("ETH")),
    })
    f = f.replace([np.inf, -np.inf], np.nan)
    f.index = trades.index
    return f[FEATURES]
