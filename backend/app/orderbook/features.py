"""Order-book measurements: spread, depth in price bands, imbalance,
rolling averages and a grouped depth ladder for display."""
from __future__ import annotations

import math
from collections import deque

from app.orderbook.book import OrderBook

# Bands measured on each side of the mid price (fractions: 0.001 = 0.1%)
BANDS = (0.001, 0.0025, 0.005, 0.01)


def band_depth(book: OrderBook, pct: float) -> tuple[float, float]:
    """Contracts resting within `pct` of mid on the bid and ask side."""
    mid = book.mid
    if mid is None:
        return 0.0, 0.0
    lo, hi = mid * (1 - pct), mid * (1 + pct)
    bid = sum(q for p, q in book.bids.items() if p >= lo)
    ask = sum(q for p, q in book.asks.items() if p <= hi)
    return bid, ask


def imbalance(bid: float, ask: float) -> float | None:
    """+1 = all resting liquidity is bids, -1 = all asks, 0 = balanced."""
    total = bid + ask
    return None if total <= 0 else (bid - ask) / total


def book_metrics(book: OrderBook, contract_size: float) -> dict:
    bb, ba, mid = book.best_bid, book.best_ask, book.mid
    coverage = book.coverage_pct()
    bands = []
    for pct in BANDS:
        b, a = band_depth(book, pct)
        bands.append({
            "pct": pct,
            "bid": b * contract_size,
            "ask": a * contract_size,
            "imbalance": imbalance(b, a),
            # False if the snapshot didn't reach this far: numbers may be understated
            "complete": coverage >= pct,
        })
    return {
        "best_bid": bb,
        "best_ask": ba,
        "mid": mid,
        "spread": None if bb is None or ba is None else ba - bb,
        "spread_bps": None if mid is None or bb is None else (ba - bb) / mid * 1e4,
        "bands": bands,
        "coverage_pct": coverage,
    }


def nice_step(raw: float) -> float:
    """Round up to 1, 2, 2.5 or 5 x 10^k."""
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    base = raw / 10**exp
    for m in (1, 2, 2.5, 5, 10):
        if base <= m + 1e-9:
            return m * 10**exp
    return 10 ** (exp + 1)


def ladder(book: OrderBook, contract_size: float, rows: int = 12, step: float | None = None) -> dict:
    """Depth grouped into price buckets of about 1 basis point."""
    mid = book.mid
    if mid is None:
        return {"step": None, "bids": [], "asks": []}
    step = step or nice_step(mid * 0.0001)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    lo, hi = mid - step * (rows + 1), mid + step * (rows + 1)
    for p, q in book.bids.items():
        if p >= lo:
            k = math.floor(p / step + 1e-9) * step
            bids[k] = bids.get(k, 0.0) + q * contract_size
    for p, q in book.asks.items():
        if p <= hi:
            k = math.ceil(p / step - 1e-9) * step
            asks[k] = asks.get(k, 0.0) + q * contract_size
    return {
        "step": step,
        "bids": [{"price": round(p, 10), "size": s} for p, s in sorted(bids.items(), reverse=True)[:rows]],
        "asks": [{"price": round(p, 10), "size": s} for p, s in sorted(asks.items())[:rows]],
    }


class RollingMean:
    """Time-windowed mean of a sampled value (samples taken about once a second)."""

    def __init__(self, max_window: float = 300.0):
        self.max_window = max_window
        self.samples: deque[tuple[float, float]] = deque()

    def add(self, t: float, value: float | None) -> None:
        if value is not None:
            self.samples.append((t, value))
        while self.samples and t - self.samples[0][0] > self.max_window:
            self.samples.popleft()

    def mean(self, window: float, now: float) -> float | None:
        vals = [v for t, v in self.samples if now - t <= window]
        return sum(vals) / len(vals) if vals else None

    def coverage(self, window: float, now: float) -> float:
        """Fraction of the window that actually has samples (0..1)."""
        if not self.samples:
            return 0.0
        oldest = next((t for t, _ in self.samples if now - t <= window), None)
        return 0.0 if oldest is None else min(1.0, (now - oldest) / window)
