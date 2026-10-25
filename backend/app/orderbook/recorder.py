"""Save order-book snapshots to disk (one row per symbol per second) so we
build our own order-book history for training later.

Files: data/orderbook/mexc_futures/<SYMBOL>/<YYYY-MM-DD>/<HHMMSS>.parquet
Each file covers about 10 minutes. Size: roughly 25-40 MB per symbol per day.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.orderbook.book import OrderBook

log = logging.getLogger(__name__)


class OrderBookRecorder:
    def __init__(self, base_dir: Path, levels: int = 25, flush_seconds: float = 600):
        self.base_dir = base_dir
        self.levels = levels
        self.flush_seconds = flush_seconds
        self.rows: dict[str, list[dict]] = {}
        self.started: dict[str, float] = {}
        self.files_written = 0

    def add(self, symbol: str, wall_time: float, book: OrderBook, metrics: dict, contract_size: float) -> None:
        bids, asks = book.top(self.levels)
        row = {
            "timestamp": datetime.fromtimestamp(wall_time, tz=timezone.utc),
            "version": book.version,
            "contract_size": contract_size,
            "best_bid": metrics["best_bid"],
            "best_ask": metrics["best_ask"],
            "mid": metrics["mid"],
            "spread": metrics["spread"],
            "bid_px": [p for p, _ in bids],
            "bid_qty": [q for _, q in bids],     # contracts
            "ask_px": [p for p, _ in asks],
            "ask_qty": [q for _, q in asks],
        }
        for b in metrics["bands"]:
            tag = f"{b['pct'] * 1e4:g}bp"
            row[f"bid_depth_{tag}"] = b["bid"]   # base asset
            row[f"ask_depth_{tag}"] = b["ask"]
            row[f"imbalance_{tag}"] = b["imbalance"]
        self.rows.setdefault(symbol, []).append(row)
        self.started.setdefault(symbol, wall_time)

    def due(self, wall_time: float) -> list[str]:
        return [s for s, t in self.started.items() if wall_time - t >= self.flush_seconds and self.rows.get(s)]

    def flush(self, symbol: str) -> Path | None:
        rows = self.rows.pop(symbol, [])
        self.started.pop(symbol, None)
        if not rows:
            return None
        first = rows[0]["timestamp"]
        path = (self.base_dir / symbol.upper() / first.strftime("%Y-%m-%d") / f"{first.strftime('%H%M%S')}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        self.files_written += 1
        log.info("Saved %d order-book snapshots -> %s", len(rows), path)
        return path

    def flush_all(self) -> None:
        for s in list(self.rows):
            try:
                self.flush(s)
            except Exception as exc:
                log.warning("Order-book recorder flush failed for %s: %s", s, exc)

    def pending(self, symbol: str) -> int:
        return len(self.rows.get(symbol, []))
