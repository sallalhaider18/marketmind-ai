"""Market orders (executed trades): who is pushing price right now.

Every MEXC trade is tagged with the aggressive side: "buy" = someone used a
market order to buy (hit the ask), "sell" = market sell (hit the bid).
From that we keep:

  * 1-minute buckets of market buying and selling, and the CVD (cumulative
    volume delta: +buys -sells, added up) for the last few hours;
  * big market orders: fills with the same millisecond and side are merged
    (one market order eating several price levels arrives as many fills) and
    kept if at least BIG_ORDER coins;
  * a plain-English reading for the last 5 and 15 minutes, comparing who is
    hitting the market with what price actually did.

Data starts when the server starts (MEXC does not give trade history).
"""
from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone

from app.adapters.base import Trade

# >>> NUMBERS YOU CAN ADJUST
BIG_ORDER = {"BTC": 10, "ETH": 100}      # a single market order this size or more is listed as "big"
MIN_ACTIVE = {"BTC": 20, "ETH": 200}     # less traded than this in 5 min -> "quiet", no reading
CONTROL_SHARE = 0.60                     # buyers (or sellers) >= 60% of market volume = "in control"
MOVE_PCT = 0.10                          # a price move of 0.10% counts as "price moved"
FLAT_PCT = 0.05                          # under 0.05% counts as "price did not move"
KEEP_MINUTES = 240


def _minute(ts: float) -> int:
    return int(ts // 60) * 60


class OrderFlow:
    def __init__(self, asset: str):
        self.asset = asset
        self.big_size = BIG_ORDER.get(asset, 0)
        self.min_active = MIN_ACTIVE.get(asset, 0)
        self.buckets: OrderedDict[int, list[float]] = OrderedDict()   # minute -> [buy, sell, last, first price]
        self.big: deque[dict] = deque(maxlen=30)
        self.cvd_base = 0.0          # CVD of minutes already dropped from the window
        self.started: float | None = None
        self._pending: dict | None = None   # fills being merged into one market order

    # -- input -------------------------------------------------------------
    def add(self, trades: list[Trade], contract_size: float) -> None:
        for t in sorted(trades, key=lambda x: x.timestamp):
            ts = t.timestamp.timestamp()
            coins = t.contracts * contract_size
            if self.started is None:
                self.started = ts
            m = _minute(ts)
            b = self.buckets.setdefault(m, [0.0, 0.0, t.price, t.price])
            b[0 if t.side == "buy" else 1] += coins
            b[2] = t.price
            p = self._pending
            if p and p["ms"] == int(ts * 1000) and p["side"] == t.side:
                p["coins"] += coins
                p["notional"] += coins * t.price
                p["last"] = t.price
            else:
                self._flush()
                self._pending = {"ms": int(ts * 1000), "side": t.side, "coins": coins,
                                 "notional": coins * t.price, "first": t.price, "last": t.price}
        while self.buckets and next(iter(self.buckets)) < _minute(self._now()) - KEEP_MINUTES * 60:
            _, (buy, sell, *_) = self.buckets.popitem(last=False)
            self.cvd_base += buy - sell

    def _flush(self) -> None:
        p, self._pending = self._pending, None
        if p and p["coins"] >= self.big_size:
            self.big.appendleft({
                "time": datetime.fromtimestamp(p["ms"] / 1000, tz=timezone.utc).isoformat(),
                "side": p["side"], "size": p["coins"],
                "price": p["notional"] / p["coins"],
                "from": p["first"], "to": p["last"],
            })

    def _now(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    # -- readings ----------------------------------------------------------
    def window(self, minutes: int, now: float | None = None) -> dict | None:
        now = now if now is not None else self._now()
        start = _minute(now) - (minutes - 1) * 60
        rows = [(m, b) for m, b in self.buckets.items() if m >= start]
        if not rows:
            return None
        buy = sum(b[0] for _, b in rows)
        sell = sum(b[1] for _, b in rows)
        # price at the start of the window = last price of the minute before it (if we have it)
        before = [b[2] for m, b in self.buckets.items() if m < start]
        # (just after start-up: use the first price seen, once there are 2+ minutes of data)
        p0 = before[-1] if before else (rows[0][1][3] if len(rows) >= 2 else None)
        p1 = rows[-1][1][2]
        change = (p1 / p0 - 1) * 100 if p0 else None
        complete = self.started is not None and self.started <= start
        return {"minutes": minutes, "bought": buy, "sold": sell, "delta": buy - sell,
                "buy_share": buy / (buy + sell) if buy + sell else None,
                "price_change_pct": change, "complete": complete}

    def read(self, w: dict | None) -> dict:
        """Turn one window into a verdict: tone up/down/warn/neutral + text."""
        if w is None or w["price_change_pct"] is None:
            return {"tone": "neutral", "label": "Collecting data",
                    "text": "Not enough trades seen yet; readings start after a few minutes."}
        total, share, ch = w["bought"] + w["sold"], w["buy_share"], w["price_change_pct"]
        if total < self.min_active * w["minutes"] / 5:
            return {"tone": "neutral", "label": "Quiet",
                    "text": "Very little market buying or selling. No clear side."}
        sell_share = 1 - share
        if ch >= MOVE_PCT and share <= 1 - CONTROL_SHARE + 0.05:
            return {"tone": "warn", "label": "Weak rise",
                    "text": f"Price is up {ch:.2f}% but market orders are mostly SELLS ({sell_share:.0%}). "
                            "The rise is not backed by real buying and may reverse."}
        if ch <= -MOVE_PCT and share >= CONTROL_SHARE - 0.05:
            return {"tone": "warn", "label": "Weak drop",
                    "text": f"Price is down {abs(ch):.2f}% but market orders are mostly BUYS ({share:.0%}). "
                            "The drop is not backed by real selling and may bounce."}
        if abs(ch) < FLAT_PCT and share >= CONTROL_SHARE + 0.05:
            return {"tone": "down", "label": "Buyers absorbed",
                    "text": f"Buyers are hitting the asks hard ({share:.0%} of volume) but price is not rising. "
                            "Big sellers are soaking it up: bearish unless price breaks higher."}
        if abs(ch) < FLAT_PCT and sell_share >= CONTROL_SHARE + 0.05:
            return {"tone": "up", "label": "Sellers absorbed",
                    "text": f"Sellers are hitting the bids hard ({sell_share:.0%} of volume) but price is not falling. "
                            "Big buyers are soaking it up: bullish unless price breaks lower."}
        if share >= CONTROL_SHARE and ch > -FLAT_PCT:
            return {"tone": "up", "label": "Buyers in control",
                    "text": f"Market buying {share:.0%} of volume and price {ch:+.2f}%. Real buyers are pushing up."}
        if sell_share >= CONTROL_SHARE and ch < FLAT_PCT:
            return {"tone": "down", "label": "Sellers in control",
                    "text": f"Market selling {sell_share:.0%} of volume and price {ch:+.2f}%. "
                            "Real sellers are pushing down."}
        return {"tone": "neutral", "label": "Balanced",
                "text": f"Buyers {share:.0%} / sellers {sell_share:.0%}, price {ch:+.2f}%. Nobody is clearly in control."}

    def view(self, now: float | None = None) -> dict:
        self._flush_old(now)
        w5, w15 = self.window(5, now), self.window(15, now)
        r5, r15 = self.read(w5), self.read(w15)
        if r5["tone"] == r15["tone"] and r5["tone"] in ("up", "down"):
            overall = {**r15, "strength": "strong",
                       "summary": ("UP" if r5["tone"] == "up" else "DOWN") + ": 5 min and 15 min agree"}
        elif r5["tone"] in ("up", "down") and r15["tone"] in ("up", "down"):
            overall = {**r5, "tone": "neutral", "strength": "mixed",
                       "summary": "MIXED: last 5 min disagrees with last 15 min"}
        else:
            pick = r5 if r5["tone"] != "neutral" else r15
            word = {"up": "Leaning UP", "down": "Leaning DOWN", "warn": "CAREFUL", "neutral": "NO CLEAR SIDE"}
            overall = {**pick, "strength": "weak", "summary": word[pick["tone"]]}
        cvd, run = [], self.cvd_base
        for m, (buy, sell, price, _) in self.buckets.items():
            run += buy - sell
            cvd.append({"time": m, "cvd": run, "price": price})
        return {"asset": self.asset, "overall": overall,
                "windows": [{**w5, "reading": r5} if w5 else None, {**w15, "reading": r15} if w15 else None],
                "cvd": cvd[-120:], "big_orders": list(self.big)[:12], "big_order_size": self.big_size,
                "since": None if self.started is None
                else datetime.fromtimestamp(self.started, tz=timezone.utc).isoformat()}

    def _flush_old(self, now: float | None) -> None:
        # a pending market order older than 1 s is complete
        if self._pending and (now if now is not None else self._now()) * 1000 - self._pending["ms"] > 1000:
            self._flush()
