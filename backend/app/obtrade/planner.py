"""Find the strongest order-book wall sitting on a strong support/resistance
level (daily + 4-hour, see levels.py) and turn it into a limit-order plan.

Long:  a big BID wall at a SUPPORT below price.
       Limit buy one price step above the wall (so it fills before the wall),
       stop just below the NEXT support down, target just below the nearest
       resistance above price.
Short: a big ASK wall at a RESISTANCE above price; mirror image.

"Big" = the largest price bucket within WALL_WINDOW of the level is at least
MIN_WALL_RATIO times the typical bucket size on that side of the book.
Only levels inside the part of the book that is fully visible are used (the
MEXC snapshot covers about +/-0.6% around BTC's price). The strongest
candidate wins (wall size x level strength); reward:risk must be >= MIN_RR.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from app.obtrade.levels import Level, split_levels
from app.orderbook.book import OrderBook
from app.orderbook.features import nice_step

WALL_WINDOW = 0.0015     # look for the wall within +/-0.15% of the level
MIN_WALL_RATIO = 4.0
MIN_RR = 1.2
BUFFER = 0.001           # stop beyond / target before the level by 0.1%


@dataclass
class Setup:
    side: str                # "long" | "short"
    entry: float             # limit price
    stop: float
    target: float
    rr: float
    level_price: float
    level_touches: int
    wall_price: float
    wall_size: float         # base asset, largest bucket at the level
    wall_total: float        # base asset, everything within the window (used to watch the wall)
    wall_ratio: float
    window_lo: float
    window_hi: float
    step: float
    stop_level: float
    target_level: float

    def as_dict(self) -> dict:
        return asdict(self)

    def reason(self, asset: str) -> str:
        side = "buy" if self.side == "long" else "sell"
        where = "support" if self.side == "long" else "resistance"
        return (f"{self.wall_size:,.2f} {asset} {side} wall ({self.wall_ratio:.0f}x a normal level) at the "
                f"strong {where} {self.level_price:,.2f} ({self.level_touches} "
                f"{'touch' if self.level_touches == 1 else 'touches'})")


def _buckets(levels: dict[float, float], step: float, up: bool, contract_size: float) -> dict[float, float]:
    out: dict[float, float] = {}
    for p, q in levels.items():
        k = (math.ceil(p / step - 1e-9) if up else math.floor(p / step + 1e-9)) * step
        out[round(k, 10)] = out.get(round(k, 10), 0.0) + q * contract_size
    return out


def window_size(levels: dict[float, float], lo: float, hi: float, contract_size: float) -> float:
    return sum(q for p, q in levels.items() if lo <= p <= hi) * contract_size


def find_setup(book: OrderBook, contract_size: float, levels: list[Level],
               min_wall_ratio: float = MIN_WALL_RATIO, min_rr: float = MIN_RR) -> tuple[Setup | None, str]:
    """Returns (best setup or None, short explanation)."""
    mid = book.mid
    if mid is None:
        return None, "order book empty"
    cov = book.coverage_pct()
    step = nice_step(mid * 0.0002)
    lo_vis, hi_vis = mid * (1 - cov), mid * (1 + cov)
    bids = _buckets({p: q for p, q in book.bids.items() if p >= lo_vis}, step, False, contract_size)
    asks = _buckets({p: q for p, q in book.asks.items() if p <= hi_vis}, step, True, contract_size)
    if len(bids) < 10 or len(asks) < 10:
        return None, "order book too thin"
    med = {"long": statistics.median(bids.values()), "short": statistics.median(asks.values())}
    sup, res = split_levels(levels, mid)
    if not sup or not res:
        return None, "no strong levels on both sides of price"

    candidates: list[tuple[float, Setup]] = []
    notes: list[str] = []
    for side, near, far_same, opposite, book_side, buckets in (
            ("long", sup, sup, res, book.bids, bids), ("short", res, res, sup, book.asks, asks)):
        s = 1 if side == "long" else -1
        for i, lv in enumerate(near):
            if abs(lv.price - mid) / mid > cov:
                break                                   # beyond what the book shows
            lo, hi = lv.price * (1 - WALL_WINDOW), lv.price * (1 + WALL_WINDOW)
            inside = {p: v for p, v in buckets.items() if lo <= p <= hi}
            if not inside:
                continue
            wall_p, wall_q = max(inside.items(), key=lambda x: x[1])
            ratio = wall_q / med[side] if med[side] > 0 else 0.0
            if ratio < min_wall_ratio:
                continue
            entry = wall_p + s * step                   # in front of the wall
            if (side == "long" and entry >= mid) or (side == "short" and entry <= mid):
                entry = wall_p
            nxt = far_same[i + 1] if i + 1 < len(far_same) else None
            if nxt is None:
                notes.append(f"{side}: no strong level beyond {lv.price:,.2f} for the stop")
                continue
            stop = nxt.price * (1 - s * BUFFER)
            target = opposite[0].price * (1 - s * BUFFER)
            risk, reward = (entry - stop) * s, (target - entry) * s
            if risk <= 0 or reward <= 0:
                continue
            rr = reward / risk
            if rr < min_rr:
                notes.append(f"{side} at {lv.price:,.2f}: reward:risk {rr:.2f} below {min_rr}")
                continue
            setup = Setup(side, entry, stop, target, rr, lv.price, lv.touches, wall_p, wall_q,
                          window_size(book_side, lo, hi, contract_size), ratio, lo, hi, step,
                          nxt.price, opposite[0].price)
            candidates.append((ratio * (1 + 0.5 * (lv.touches - 1)), setup))
    if not candidates:
        why = "; ".join(notes[:2]) if notes else (
            f"no wall >= {min_wall_ratio:.0f}x normal size on a strong level within ±{cov * 100:.2f}% (visible book)")
        return None, why
    best = max(candidates, key=lambda x: x[0])[1]
    return best, "ok"
