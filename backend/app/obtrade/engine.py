"""Order-book trade plans, checked every 30 minutes during the Asia, London and
New York sessions (every day), with a paper record of every plan.

Per coin there is at most one plan at a time:
  pending  -> a paper LIMIT order is waiting at the entry price
              cancelled if: the wall behind it shrinks below 40% of its size
              ("wall pulled"), price reaches the target without filling ("missed"),
              or it has not filled within PENDING_HOURS ("expired")
  open     -> filled; closed at the target, the stop, or after MAX_HOLD_DAYS
Fills, stops and targets are checked every few seconds against every trade
printed on MEXC since the last check (if one check sees both stop and target,
the stop counts). Paper account: $10,000, 1% risk per trade; limit entries and
targets pay 0% (maker), stops and time exits 0.02% (taker).
No real orders are ever placed.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from app.obtrade.levels import find_levels, split_levels
from app.obtrade.planner import find_setup, window_size

log = logging.getLogger("marketmind.plan")

SESSIONS = {             # local opening hours, every day
    "Asia": ("Asia/Tokyo", 9, 18),
    "London": ("Europe/London", 8, 17),
    "New York": ("America/New_York", 8, 17),
}
PENDING_HOURS = 12
MAX_HOLD_DAYS = 3
WALL_PULLED = 0.4
TAKER = 0.0002


def active_sessions(now: datetime) -> list[str]:
    out = []
    for name, (tz, start, end) in SESSIONS.items():
        h = pd.Timestamp(now).tz_convert(tz).hour
        if start <= h < end:
            out.append(name)
    return out


def next_check_time(now: datetime) -> datetime:
    """Next :00 or :30 (UTC) at which a session is open."""
    t = pd.Timestamp(now).floor("30min") + pd.Timedelta(minutes=30)
    for _ in range(100):
        if active_sessions(t.to_pydatetime()):
            return t.to_pydatetime()
        t += pd.Timedelta(minutes=30)
    return t.to_pydatetime()


def fmt(now: datetime) -> str:
    ts = pd.Timestamp(now)
    return f"{ts.tz_convert('Europe/London'):%a %H:%M} UK / {ts.tz_convert('America/New_York'):%H:%M} NY"


class PlanEngine:
    def __init__(self, live, contract_size: Callable[[str], float | None],
                 candles: Callable[[str, str], pd.DataFrame], state_path: Path, log_path: Path,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 out: Callable[[str], None] = log.info, risk_pct: float = 0.01):
        self.live, self.contract_size, self.candles = live, contract_size, candles
        self.state_path, self.log_path, self.clock, self.say, self.risk_pct = state_path, log_path, clock, out, risk_pct
        self.symbols = list(live.symbols)
        self.levels: dict[str, list] = {s: [] for s in self.symbols}
        self.last_msg: dict[str, str] = {s: "waiting for the first check" for s in self.symbols}
        self.last_check: datetime | None = None
        self.next_check: datetime | None = None
        self._seen: dict[str, float] = {}
        self._retry: set[str] = set()          # checks skipped because the book was not ready
        self.st = self._load()

    # ---------------------------------------------------------------- state --
    def _load(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                pass
        return {"equity": 10_000.0, "closed": 0, "wins": 0, "total_r": 0.0,
                "active": {}, "history": []}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.st, indent=2, default=str))

    def _log(self, row: dict) -> None:
        new = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def _archive(self, plan: dict) -> None:
        self.st["history"] = ([plan] + self.st["history"])[:30]

    # -------------------------------------------------------------- prices --
    def _range_since_last(self, sym: str) -> tuple[float, float, float] | None:
        """(high, low, last) of trades since the previous call, or None if nothing traded."""
        st = self.live.symbols[sym]
        since = self._seen.get(sym, 0.0)
        prices = [t.price for t in st.trades if t.timestamp.timestamp() > since]
        if st.trades:
            self._seen[sym] = st.trades[-1].timestamp.timestamp()
        mid = st.book.book.mid if st.book.synced else None
        if not prices:
            return (mid, mid, mid) if mid else None
        last = prices[-1]
        return max(prices), min(prices), last

    # ------------------------------------------------------------ main loop --
    async def run(self, every: float = 5.0) -> None:
        while True:
            try:
                await self.step()
            except Exception:
                log.exception("plan engine step failed")
            await asyncio.sleep(every)

    async def step(self) -> None:
        now = self.clock()
        for sym in self.symbols:
            self._monitor(sym, now)
        if self.next_check is None:
            self.next_check = next_check_time(now)
            if active_sessions(now) and self.last_check is None:
                self.next_check = now           # run a first check straight away during a session
        due: list[str] = []
        if now >= self.next_check:
            self.last_check = now
            self.next_check = next_check_time(now)
            due = list(self.symbols)
            self._retry.clear()
        elif self._retry:
            due = [s for s in list(self._retry) if self._book_ready(s)]   # book is ready now: run the skipped check
        for sym in due:
            self._retry.discard(sym)
            try:
                df4 = await asyncio.to_thread(self.candles, sym, "4h")
                df1 = await asyncio.to_thread(self.candles, sym, "1d")
                book = self.live.symbols[sym].book
                mid = book.book.mid if book.synced else None
                self.levels[sym] = find_levels(df4, df1, current=mid)
            except Exception as exc:
                self.say(f"{sym}: could not load 4h / daily candles ({type(exc).__name__}: {exc})")
            self._check(sym, now)
        self._save()

    def _book_ready(self, sym: str) -> bool:
        st = self.live.symbols[sym]
        return st.book.synced and not self.live.is_stale(sym) and self.contract_size(sym) is not None

    # --------------------------------------------------------- 30-min check --
    def _check(self, sym: str, now: datetime) -> None:
        st = self.live.symbols[sym]
        asset = sym.replace("USDT.P", "")
        sessions = active_sessions(now)
        plan = self.st["active"].get(sym)
        cs = self.contract_size(sym)
        if not self._book_ready(sym):
            self.last_msg[sym] = "waiting for the order book to sync"
            if sym not in self._retry:
                self.say(f"{fmt(now)}  {sym:<10} order book not ready yet: will check as soon as it is")
            self._retry.add(sym)
            return
        book = st.book.book
        if plan and plan["status"] == "pending":
            side_book = book.bids if plan["side"] == "long" else book.asks
            now_size = window_size(side_book, plan["window_lo"], plan["window_hi"], cs)
            if now_size < WALL_PULLED * plan["wall_total"]:
                self._cancel(sym, plan, now, f"wall pulled ({now_size:,.2f} {asset} left of "
                                              f"{plan['wall_total']:,.2f})")
            else:
                self.last_msg[sym] = (f"limit {('buy' if plan['side'] == 'long' else 'sell')} waiting at "
                                      f"{plan['entry']:,.2f}; wall still there ({now_size:,.2f} {asset})")
                self.say(f"{fmt(now)}  {sym:<10} {self.last_msg[sym]}")
            return
        if plan and plan["status"] == "open":
            self.last_msg[sym] = f"{plan['side']} trade open from {plan['fill_price']:,.2f}"
            self.say(f"{fmt(now)}  {sym:<10} {self.last_msg[sym]}")
            return
        if not sessions:
            return
        setup, why = find_setup(book, cs, self.levels[sym])
        if setup is None:
            self.last_msg[sym] = f"no setup: {why}"
            self.say(f"{fmt(now)}  {sym:<10} [{' + '.join(sessions)}] no setup: {why}")
            return
        plan = {**setup.as_dict(), "symbol": sym, "asset": asset, "status": "pending",
                "placed_at": now.isoformat(), "expires_at": (now + timedelta(hours=PENDING_HOURS)).isoformat(),
                "session": " + ".join(sessions), "why": setup.reason(asset)}
        self.st["active"][sym] = plan
        word = "BUY " if setup.side == "long" else "SELL"
        self.last_msg[sym] = f"limit {word.strip().lower()} placed at {setup.entry:,.2f}"
        self.say(f"{fmt(now)}  {sym:<10} >>> PLACE LIMIT {word} at {setup.entry:,.2f}   stop {setup.stop:,.2f}   "
                 f"target {setup.target:,.2f}   reward:risk {setup.rr:.2f}   because: {plan['why']}")

    # ------------------------------------------------------------- monitor --
    def _monitor(self, sym: str, now: datetime) -> None:
        plan = self.st["active"].get(sym)
        rng = self._range_since_last(sym)
        if not plan or rng is None:
            return
        hi, lo, last = rng
        s = 1 if plan["side"] == "long" else -1
        if plan["status"] == "pending":
            filled = (lo <= plan["entry"]) if s == 1 else (hi >= plan["entry"])
            if filled:
                plan.update(status="open", fill_price=plan["entry"], filled_at=now.isoformat())
                self.say(f"{fmt(now)}  {sym:<10} === FILLED {plan['side'].upper()} at {plan['entry']:,.2f}   "
                         f"stop {plan['stop']:,.2f}   target {plan['target']:,.2f}")
                if (s == 1 and lo <= plan["stop"]) or (s == -1 and hi >= plan["stop"]):
                    self._close(sym, plan, plan["stop"], "stop", now, taker=True)
                return
            if (s == 1 and hi >= plan["target"]) or (s == -1 and lo <= plan["target"]):
                self._cancel(sym, plan, now, "price reached the target before the order filled")
            elif now >= datetime.fromisoformat(plan["expires_at"]):
                self._cancel(sym, plan, now, f"not filled within {PENDING_HOURS} h")
            return
        if plan["status"] == "open":
            if (s == 1 and lo <= plan["stop"]) or (s == -1 and hi >= plan["stop"]):
                self._close(sym, plan, plan["stop"], "stop", now, taker=True)
            elif (s == 1 and hi >= plan["target"]) or (s == -1 and lo <= plan["target"]):
                self._close(sym, plan, plan["target"], "target", now, taker=False)
            elif now >= datetime.fromisoformat(plan["filled_at"]) + timedelta(days=MAX_HOLD_DAYS):
                self._close(sym, plan, last, f"{MAX_HOLD_DAYS}-day limit", now, taker=True)

    def _cancel(self, sym: str, plan: dict, now: datetime, reason: str) -> None:
        plan.update(status="cancelled", ended_at=now.isoformat(), result=reason)
        del self.st["active"][sym]
        self._archive(plan)
        self.last_msg[sym] = f"order cancelled: {reason}"
        self.say(f"{fmt(now)}  {sym:<10} xxx CANCELLED limit {plan['side']} at {plan['entry']:,.2f}: {reason}")
        self._log({"symbol": sym, "side": plan["side"], "status": "cancelled", "placed_at": plan["placed_at"],
                   "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"], "rr": round(plan["rr"], 2),
                   "filled_at": "", "exit_time": now.isoformat(), "exit": "", "reason": reason, "r": "", "pnl_usd": "",
                   "equity": round(self.st["equity"], 2), "why": plan["why"]})

    def _close(self, sym: str, plan: dict, price: float, reason: str, now: datetime, taker: bool) -> None:
        s = 1 if plan["side"] == "long" else -1
        risk_usd = self.st["equity"] * self.risk_pct
        qty = risk_usd / abs(plan["entry"] - plan["stop"])
        fees = price * qty * (TAKER if taker else 0.0)
        pnl = s * (price - plan["entry"]) * qty - fees
        r = pnl / risk_usd
        self.st["equity"] += pnl
        self.st["closed"] += 1
        self.st["wins"] += int(pnl > 0)
        self.st["total_r"] += r
        plan.update(status="closed", ended_at=now.isoformat(), exit=price, result=reason, r=r, pnl_usd=pnl)
        del self.st["active"][sym]
        self._archive(plan)
        word = "WIN " if pnl > 0 else "LOSS"
        self.last_msg[sym] = f"trade closed: {reason}, {r:+.2f} R"
        self.say(f"{fmt(now)}  {sym:<10} <<< CLOSED {plan['side'].upper()} at {price:,.2f} ({reason})   {word} "
                 f"{r:+.2f} R  ${pnl:+,.2f}   paper equity ${self.st['equity']:,.2f}   "
                 f"[{self.st['wins']}/{self.st['closed']} wins, total {self.st['total_r']:+.2f} R]")
        self._log({"symbol": sym, "side": plan["side"], "status": "closed", "placed_at": plan["placed_at"],
                   "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"], "rr": round(plan["rr"], 2),
                   "filled_at": plan.get("filled_at", ""), "exit_time": now.isoformat(), "exit": price,
                   "reason": reason, "r": round(r, 3), "pnl_usd": round(pnl, 2),
                   "equity": round(self.st["equity"], 2), "why": plan["why"]})

    # ---------------------------------------------------------------- view --
    def view(self, sym: str) -> dict:
        now = self.clock()
        book = self.live.symbols[sym].book.book
        mid = book.mid
        lv = self.levels.get(sym, [])
        shown = []
        if mid:
            sup, res = split_levels(lv, mid)
            shown = [x.as_dict(mid) for x in sup + res]
        return {
            "sessions": active_sessions(now),
            "next_check": self.next_check.isoformat() if self.next_check else None,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "message": self.last_msg.get(sym),
            "active": self.st["active"].get(sym),
            "history": [h for h in self.st["history"] if h["symbol"] == sym][:8],
            "levels": shown,
            "account": {k: self.st[k] for k in ("equity", "closed", "wins", "total_r")},
        }
