"""Turn order-book observations into a few plain-English alerts.

Only the strongest bid and the strongest ask near the price are reported
(one at a time per side), plus what happens to them later (pulled, filled,
absorbing).

An order must be the biggest on its side, hold 30 s and be at least MIN_WALL
coins (100 BTC / 1,000 ETH). When price reaches it we say which of three
things happened: pulled (fake), eaten and price broke through, or eaten
without a break.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from datetime import datetime, timezone

from app.orderbook.large_orders import TrackedOrder

HOLD_SECONDS = 30            # a large order must stay this long to be reported
NEAR_PRICE = 0.005           # only report large orders within 0.5% of price
END_NEAR_PRICE = 0.003       # report pulls/fills only if they happened within 0.3%
IMBALANCE_LEVEL = 0.25       # |imbalance| needed on both 15 s and 60 s averages
FLOW_CONFIRM = 0.60          # share of aggressive volume on the same side
COOLDOWN = 120               # seconds before the same alert key can repeat
# >>> THE NUMBER TO ADJUST: minimum wall size in coins. Too many alerts -> raise it; too few -> lower it.
MIN_WALL = {"BTC": 100, "ETH": 1000}
EATEN_SHARE = 0.5            # at least half the wall must trade for it to count as "eaten"
BREAK_PCT = 0.0005           # price must move 0.05% past the wall ...
BREAK_WINDOW = 120           # ... within 2 minutes to count as "broke through"
STRONG_REPEAT = 600          # per side: at most one new "strongest" alert every 10 min ...
BIGGER_X = 1.5               # ... unless the new one is at least 1.5x bigger

_ids = itertools.count(1)


class AlertEngine:
    def __init__(self, symbol: str, base_asset: str, price_fmt, min_wall: float | None = None):
        self.min_wall = MIN_WALL.get(base_asset, 0) if min_wall is None else min_wall   # coins
        self.symbol = symbol
        self.asset = base_asset
        self.fmt = price_fmt
        self.recent: deque[dict] = deque(maxlen=40)
        self._last: dict[str, float] = {}
        self._strong: dict[str, tuple[float, float]] = {}   # side -> (time, size) of last strongest alert
        self._watch: list[TrackedOrder] = []   # eaten walls waiting to see if price breaks through

    def _emit(self, key: str, kind: str, tone: str, text: str, now: float) -> None:
        if now - self._last.get(key, -1e9) < COOLDOWN:
            return
        self._last[key] = now
        self.recent.appendleft({
            "id": next(_ids),
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "kind": kind,
            "tone": tone,
            "text": text,
        })

    def _group_text(self, group: list[TrackedOrder], noun: str, verb: str, mid: float, cs: float,
                    where: str | None) -> str:
        prices = sorted(o.price for o in group)
        sizes = [o.max_size for o in group]
        text = (f"{len(group)} {noun} between {self.fmt(prices[0])} and {self.fmt(prices[-1])} {verb}"
                f" ({self._size(sum(sizes), cs)} in total")
        if where:
            nearest = min(abs(p - mid) for p in prices) / mid
            text += f", nearest {nearest * 100:.2f}% {where} price"
        text += ")."
        if max(sizes) <= 1.1 * min(sizes):
            text += " Their sizes are almost identical, which usually means one participant's quote ladder."
        return text

    def _size(self, contracts: float, cs: float) -> str:
        return f"{contracts * cs:,.2f} {self.asset}"

    def evaluate(self, now: float, mid: float, cs: float, active: list[TrackedOrder],
                 ended: list[TrackedOrder], imb15: float | None = None, imb60: float | None = None,
                 imb_coverage: float = 0, buy_ratio: float | None = None,
                 threshold: float | None = None) -> None:
        """Only the single strongest bid and strongest ask are ever reported.

        The strongest order must be the biggest on its side within 0.5% of price,
        held 30 s and at least min_wall coins. Book-pressure alerts are not produced."""
        for side in ("bid", "ask"):
            near = [o for o in active if o.side == side and abs(o.price - mid) / mid <= NEAR_PRICE]
            if not near:
                continue
            # the biggest order on this side RIGHT NOW (same as the top of the "large visible
            # orders" list). If it hasn't held 30 s yet we wait - we never fall back to a smaller one.
            o = max(near, key=lambda x: x.size)
            if o.age(now) < HOLD_SECONDS or "hold" in o.flags:
                continue
            if o.size * cs < self.min_wall:
                continue
            last_t, last_size = self._strong.get(side, (-1e9, 0.0))
            if now - last_t < STRONG_REPEAT and o.size < BIGGER_X * last_size:
                continue      # recently reported one; only a clearly bigger order replaces it
            o.flags.add("hold")
            self._strong[side] = (now, o.size)
            word = "bid" if side == "bid" else "ask"
            where = "below" if side == "bid" else "above"
            self._emit(f"hold:{side}:{o.price}", "large_order", "up" if side == "bid" else "down",
                       f"Strongest {word}: {self._size(o.size, cs)} ({_usd(o.size * cs * mid)}) at {self.fmt(o.price)}, "
                       f"{abs(o.price - mid) / mid * 100:.2f}% {where} price, held {int(o.age(now))} s.", now)

        for o in active:
            if "hold" in o.flags and o.status(now) == "absorbing" and "absorb" not in o.flags:
                o.flags.add("absorb")
                word = "bid" if o.side == "bid" else "ask"
                taker = "Sellers" if o.side == "bid" else "Buyers"
                self._emit(f"absorb:{o.side}:{o.price}", "absorption", "up" if o.side == "bid" else "down",
                           f"{taker} traded {self._size(o.hit, cs)} into the strongest {word} at "
                           f"{self.fmt(o.price)} and it is still there. Possible absorption.", now)

        # Only follow up on orders we already reported as holding; quotes that
        # come and go within seconds are routine (market makers repositioning).
        reportable = [o for o in ended if "hold" in o.flags and o.distance_at_end is not None
                      and o.distance_at_end <= END_NEAR_PRICE]
        # Several orders on one side ending the same way at the same moment: one alert.
        groups: dict[tuple[str, str], list[TrackedOrder]] = {}
        for o in reportable:
            groups.setdefault((o.side, o.end_reason), []).append(o)
        singles = []
        for (side, reason), group in groups.items():
            if len(group) == 1:
                singles.append(group[0])
                continue
            word = "bid" if side == "bid" else "ask"
            what = ("were removed before price reached them" if reason == "pulled"
                    else "were reached by price and are gone")
            self._emit(f"{reason}:{side}:group:{min(o.price for o in group)}", reason,
                       "neutral" if reason == "pulled" else ("down" if side == "bid" else "up"),
                       self._group_text(group, f"large {word}s", what, mid, cs, None), now)

        for o in singles:
            word = "bid" if o.side == "bid" else "ask"
            if o.end_reason == "pulled":
                self._emit(f"pulled:{o.side}:{o.price}", "pulled", "neutral",
                           f"Strongest {word} at {self.fmt(o.price)} ({self._size(o.max_size, cs)}) was removed "
                           f"before price reached it. Probably fake - don't trust that level.", now)
            elif o.hit >= EATEN_SHARE * o.max_size:
                self._watch.append(o)          # eaten: wait to see whether price breaks through
            else:
                self._emit(f"filled:{o.side}:{o.price}", "pulled", "neutral",
                           f"Strongest {word} at {self.fmt(o.price)} ({self._size(o.max_size, cs)}) disappeared "
                           f"as price arrived, but only {self._size(o.hit, cs)} traded into it. "
                           f"Probably pulled at the last moment.", now)

        self._check_breaks(now, mid, cs)

    def _check_breaks(self, now: float, mid: float, cs: float) -> None:
        keep = []
        for o in self._watch:
            word = "bid" if o.side == "bid" else "ask"
            broke = (mid >= o.price * (1 + BREAK_PCT)) if o.side == "ask" else (mid <= o.price * (1 - BREAK_PCT))
            if broke:
                if o.side == "ask":
                    text = (f"BROKE THROUGH: strongest ask of {self._size(o.max_size, cs)} at {self.fmt(o.price)} "
                            f"was eaten ({self._size(o.hit, cs)} traded) and price moved above it. "
                            f"Buyers were strong enough - possible move up.")
                else:
                    text = (f"BROKE THROUGH: strongest bid of {self._size(o.max_size, cs)} at {self.fmt(o.price)} "
                            f"was eaten ({self._size(o.hit, cs)} traded) and price moved below it. "
                            f"Sellers were strong enough - possible move down.")
                self._emit(f"break:{o.side}:{o.price}", "breakout", "up" if o.side == "ask" else "down", text, now)
            elif now - (o.ended_at or now) > BREAK_WINDOW:
                self._emit(f"filled:{o.side}:{o.price}", "filled", "neutral",
                           f"Strongest {word} at {self.fmt(o.price)} was eaten ({self._size(o.hit, cs)} traded) "
                           f"but price did not break through within 2 minutes.", now)
            else:
                keep.append(o)
        self._watch = keep


def _usd(x: float) -> str:
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k"
