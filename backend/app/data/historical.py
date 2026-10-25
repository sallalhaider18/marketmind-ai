"""Download historical candles and keep a local Parquet copy up to date.

Files live at data/raw/<venue>/<SYMBOL>/<interval>.parquet, e.g.
data/raw/mexc_futures/BTCUSDT.P/1m.parquet
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.adapters.base import CANDLE_COLUMNS, INTERVAL_SECONDS, MarketDataAdapter

log = logging.getLogger(__name__)


def candle_path(raw_dir: Path, venue: str, symbol: str, interval: str) -> Path:
    return raw_dir / venue / symbol.upper() / f"{interval}.parquet"


def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def merge_candles(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combine, newest download wins on duplicate timestamps, sorted ascending."""
    frames = [f for f in (existing, new) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return (
        out.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)[CANDLE_COLUMNS]
    )


def last_closed_candle_end(interval: str, now: datetime | None = None) -> datetime:
    """Open time of the currently-forming candle; download up to (not including) it."""
    now = now or datetime.now(timezone.utc)
    step = INTERVAL_SECONDS[interval]
    return datetime.fromtimestamp(int(now.timestamp()) // step * step, tz=timezone.utc)


def update_history(
    adapter: MarketDataAdapter,
    symbol: str,
    interval: str,
    raw_dir: Path,
    days: int = 30,
    now: datetime | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Ensure the local file covers the last `days` days of *closed* candles.

    Only missing ranges are downloaded: earlier history if `days` grew, and
    everything after the last stored candle.
    """
    path = candle_path(raw_dir, adapter.venue, symbol, interval)
    existing = load_candles(path)
    end = last_closed_candle_end(interval, now)
    start = end - timedelta(days=days)
    step = timedelta(seconds=INTERVAL_SECONDS[interval])

    pieces = []
    if existing.empty:
        log.info("%s %s: downloading %s -> %s", symbol, interval, start, end)
        pieces.append(adapter.fetch_candles(symbol, interval, start, end))
    else:
        first = existing["timestamp"].iloc[0].to_pydatetime()
        last = existing["timestamp"].iloc[-1].to_pydatetime()
        if start < first:
            log.info("%s %s: back-filling %s -> %s", symbol, interval, start, first)
            pieces.append(adapter.fetch_candles(symbol, interval, start, first))
        if last + step < end:
            log.info("%s %s: updating %s -> %s", symbol, interval, last + step, end)
            pieces.append(adapter.fetch_candles(symbol, interval, last + step, end))

    merged = existing
    for p in pieces:
        merged = merge_candles(merged, p)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: never leaves a half-written file
    return path, merged
