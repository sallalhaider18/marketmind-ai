"""Track unusually large visible orders near the price.

A level counts as "large" when its size is at least `multiple` times the
median level size within `band_pct` of mid (capped at the range the order-book
snapshot covers). Each large level is followed over
time: how long it stays, whether it grows or shrinks, how much was traded
into it, and how it ended:
  - "filled":  price reached it (trades at/through its price) before it went
  - "pulled":  it disappeared while price was still away from it
Neither outcome says who placed it or why.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

from app.adapters.base import Trade
from app.orderbook.book import OrderBook


@dataclass
class TrackedOrder:
    side: str                 # "bid" or "ask"
    price: float
    first_seen: float
    last_seen: float
    initial: float            # contracts
    size: float
    max_size: float
    hit: float = 0.0          # contracts traded into this level by aggressive orders
    touched_at: float | None = None
    missing_since: float | None = None
    ended_at: float | None = None
    end_reason: str | None = None   # "filled" | "pulled"
    distance_at_end: float | None = None
    flags: set = field(default_factory=set)  # alerts already sent

    @property
    def key(self) -> tuple[str, float]:
        return (self.side, self.price)

    def age(self, now: float) -> float:
        return (self.ended_at or now) - self.first_seen

    def status(self, now: float) -> str:
        if self.end_reason:
            return self.end_reason
        if self.age(now) < 10:
            return "new"
        if self.hit >= 0.25 * self.max_size:
            return "absorbing"
        if self.size > 1.25 * self.initial:
            return "growing"
        if self.size < 0.75 * self.initial:
            return "shrinking"
        return "holding"


class LargeOrderTracker:
    def __init__(self, band_pct: float = 0.005, multiple: float = 8.0, max_per_side: int = 6,
                 grace_seconds: float = 2.0):
        self.band_pct = band_pct
        self.multiple = multiple
        self.max_per_side = max_per_side
        self.grace = grace_seconds
        self.active: dict[tuple[str, float], TrackedOrder] = {}
        self.ended: deque[TrackedOrder] = deque(maxlen=30)
        self.threshold: float | None = None
        self.typical: float | None = None
        self.band_used: float = band_pct

    def _candidates(self, book: OrderBook) -> dict[tuple[str, float], float]:
        mid = book.mid
        # Stay inside the range the snapshot covers; beyond it the book may be incomplete.
        cov = book.coverage_pct()
        band = min(self.band_pct, cov) if cov > 0 else self.band_pct
        self.band_used = band
        lo, hi = mid * (1 - band), mid * (1 + band)
        bids = [(p, q) for p, q in book.bids.items() if p >= lo]
        asks = [(p, q) for p, q in book.asks.items() if p <= hi]
        sizes = [q for _, q in bids + asks]
        if len(sizes) < 6:
            self.threshold = None
            return {}
        self.typical = statistics.median(sizes)
        self.threshold = self.typical * self.multiple
        out = {}
        for side, levels in (("bid", bids), ("ask", asks)):
            big = sorted((lv for lv in levels if lv[1] >= self.threshold), key=lambda x: -x[1])
            for p, q in big[: self.max_per_side]:
                out[(side, p)] = q
        return out

    def update(self, now: float, book: OrderBook, trades: list[Trade]) -> list[TrackedOrder]:
        """Advance one sample. Returns orders that ended during this update."""
        if book.mid is None:
            return []
        mid = book.mid
        candidates = self._candidates(book)
        if self.threshold is None:
            return []

        # Trades: credit volume to the level it hit, and note when price reached a level.
        for o in self.active.values():
            for t in trades:
                at_level = abs(t.price - o.price) <= o.price * 1e-9
                if o.side == "bid" and t.side == "sell" and t.price <= o.price + o.price * 1e-9:
                    o.touched_at = now
                    if at_level:
                        o.hit += t.contracts
                elif o.side == "ask" and t.side == "buy" and t.price >= o.price - o.price * 1e-9:
                    o.touched_at = now
                    if at_level:
                        o.hit += t.contracts

        ended = []
        for key, o in list(self.active.items()):
            levels = book.bids if o.side == "bid" else book.asks
            q = levels.get(o.price, 0.0)
            through = (o.side == "bid" and book.best_ask is not None and book.best_ask <= o.price) or \
                      (o.side == "ask" and book.best_bid is not None and book.best_bid >= o.price)
            if through:
                o.touched_at = now
            # still there: at least half the large size AND at least a quarter of its peak
            # (an order that shrinks to a sliver is treated as gone)
            if q >= max(0.5 * self.threshold, 0.25 * o.max_size):
                o.size, o.last_seen, o.missing_since = q, now, None
                o.max_size = max(o.max_size, q)
                continue
            if o.missing_since is None:
                o.missing_since = now
            if now - o.missing_since >= self.grace:
                o.ended_at = o.missing_since
                recently_touched = o.touched_at is not None and o.touched_at >= o.last_seen - 2
                o.end_reason = "filled" if recently_touched or through else "pulled"
                o.distance_at_end = abs(o.price - mid) / mid
                o.size = q
                del self.active[key]
                self.ended.appendleft(o)
                ended.append(o)

        for key, q in candidates.items():
            if key not in self.active:
                self.active[key] = TrackedOrder(side=key[0], price=key[1], first_seen=now, last_seen=now,
                                                initial=q, size=q, max_size=q)
        return ended

    def view(self, now: float, mid: float | None, contract_size: float) -> dict:
        def row(o: TrackedOrder) -> dict:
            return {
                "side": o.side,
                "price": o.price,
                "size": o.size * contract_size,
                "max_size": o.max_size * contract_size,
                "traded_into": o.hit * contract_size,
                "age_s": round(o.age(now), 1),
                "status": o.status(now),
                "distance_pct": None if mid is None else (o.price - mid) / mid * 100,
                "times_typical": None if not self.typical else o.max_size / self.typical,
            }
        active = sorted(self.active.values(), key=lambda o: -o.size)
        return {
            "threshold": None if self.threshold is None else self.threshold * contract_size,
            "band_pct": self.band_used,
            "typical_level": None if self.typical is None else self.typical * contract_size,
            "active": [row(o) for o in active],
            "recent": [row(o) for o in list(self.ended)[:8]],
        }
