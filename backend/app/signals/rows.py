"""One row per symbol per second: everything the signal engine needs.

Rows are saved to disk so the exact live inputs can be replayed later:
data/signals/mexc_futures/<SYMBOL>/<YYYY-MM-DD>/<HHMMSS>.parquet (10 min each,
about 2-4 MB per symbol per day).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
WALL_RANGE = 0.005      # walls within 0.5% of price are stored
WALLS_PER_SIDE = 5


def build_row(t: float, valid: bool, metrics: dict | None, trades: list, cs: float | None,
              active_orders: list, tracker_now: float) -> dict:
    """metrics = book_metrics(...) (bands[0] = 0.1%, bands[1] = 0.25%); active_orders = TrackedOrder list."""
    row = {"t": t, "valid": bool(valid and metrics and metrics.get("mid") and cs)}
    mid = metrics.get("mid") if metrics else None
    prices = [x.price for x in trades]
    buy = sum(x.contracts for x in trades if x.side == "buy") * (cs or 0)
    sell = sum(x.contracts for x in trades if x.side != "buy") * (cs or 0)
    last = prices[-1] if prices else mid
    row.update({
        "mid": mid,
        "bid": metrics.get("best_bid") if metrics else None,
        "ask": metrics.get("best_ask") if metrics else None,
        "last": last,
        "hi": max(prices + ([mid] if mid else [])) if (prices or mid) else None,
        "lo": min(prices + ([mid] if mid else [])) if (prices or mid) else None,
        "buy": buy, "sell": sell,
        "imb10": metrics["bands"][0]["imbalance"] if metrics else None,
        "imb25": metrics["bands"][1]["imbalance"] if metrics else None,
    })
    for side in ("bid", "ask"):
        walls = []
        if mid and cs:
            for o in active_orders:
                if o.side == side and abs(o.price - mid) / mid <= WALL_RANGE:
                    walls.append({"price": o.price, "coins": o.size * cs, "peak": o.max_size * cs,
                                  "age": o.age(tracker_now), "hit": o.hit * cs})
        walls.sort(key=lambda w: -w["coins"])
        row[f"walls_{side}"] = walls[:WALLS_PER_SIDE]
    return row


class RowRecorder:
    def __init__(self, base_dir: Path, flush_seconds: float = 600):
        self.base_dir = base_dir
        self.flush_seconds = flush_seconds
        self.rows: dict[str, list[dict]] = {}
        self.started: dict[str, float] = {}
        self.files_written = 0

    def add(self, symbol: str, row: dict) -> None:
        self.rows.setdefault(symbol, []).append(row)
        self.started.setdefault(symbol, time.time())

    def due(self, now: float) -> list[str]:
        return [s for s, t0 in self.started.items() if now - t0 >= self.flush_seconds and self.rows.get(s)]

    def flush(self, symbol: str) -> None:
        rows = self.rows.pop(symbol, [])
        self.started.pop(symbol, None)
        if not rows:
            return
        df = pd.DataFrame(rows)
        for c in ("walls_bid", "walls_ask"):
            df[c] = df[c].map(json.dumps)
        t0 = datetime.fromtimestamp(rows[0]["t"], tz=timezone.utc)
        path = self.base_dir / symbol / t0.strftime("%Y-%m-%d") / f"{t0.strftime('%H%M%S')}.parquet"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            self.files_written += 1
        except Exception:
            log.exception("could not save signal rows for %s", symbol)

    def flush_all(self) -> None:
        for s in list(self.rows):
            self.flush(s)


def load_rows(base_dir: Path, symbol: str) -> list[dict]:
    files = sorted((base_dir / symbol).glob("*/*.parquet"))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("t")
    df = df.drop_duplicates("t", keep="last")
    rows = df.to_dict("records")
    for r in rows:
        for c in ("walls_bid", "walls_ask"):
            r[c] = json.loads(r[c]) if isinstance(r[c], str) else (r[c] or [])
        for k in ("mid", "bid", "ask", "last", "hi", "lo", "imb10", "imb25"):
            if r.get(k) is not None and pd.isna(r[k]):
                r[k] = None
    return rows
