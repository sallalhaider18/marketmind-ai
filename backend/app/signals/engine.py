"""Unified order-book signal engine (one per symbol), with break-vs-reversal classification.

Input: one "row" per second (see rows.py). The live server and the replay
script feed it the same rows, so a replay tests exactly what ran live.

A strong daily/4H level becomes a SETUP when price comes within 0.15% and
touches it. At every completed 5-minute close after the touch, price decides
which of four candidates applies (anything else is "uncertain" = no trade):

  support    close above level                          -> BOUNCE candidate      (long)
             close 0.1% below level, previous close
             also below (failed reclaim)                -> BREAKDOWN candidate   (short)
  resistance close below level                          -> REJECTION candidate   (short)
             close 0.1% above level, previous close
             also above                                 -> BREAKOUT candidate    (long)

The candidate is then checked (every check is logged, passed or not):
  wall      reversals: a large order defending the level, >= 60 s old, still there;
            shrinking is fine if it shrank because it was TRADED INTO (absorbing),
            not if it was cancelled.  breaks: NO large defending order left
            (it was consumed or pulled).
  imbalance 60 s average in the trade direction >= 0.25 in the 0.1% and 0.25% ranges
  flow      aggressive orders >= 60% in the trade direction over 60 s
  cvd       reversals: last completed minute agrees; breaks: last 5 minutes agree
  risk      reward:risk >= 1.5 after fees and slippage (per strategy)
All true -> SIGNAL -> strategy A (enter now) and B (limit at the retest),
each trade tagged with its setup type.

A 30-min close through the level switches off the reversal candidate; a 30-min
close back on the original side switches off the break candidate. A setup
lasts at most 12 five-minute closes (60 min) after the touch.

Separately, every touch is followed for 60 min and saved as a LEVEL TEST:
did price move 0.3% beyond the level first (broke), 0.3% back away first
(held), or neither (unresolved)? With CVD, flow, imbalance and wall fate at
the touch, plus max excursion both ways. This is the labelled data needed to
learn later how often a level breaks under given conditions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from app.signals.rules import Rules
from app.signals.sim import Position, open_position, plan_rr, summarize, update_position

BRANCHES = {"support": (("bounce", "long"), ("breakdown", "short")),
            "resistance": (("rejection", "short"), ("breakout", "long"))}
SETUP_NAMES = {"bounce": "support bounce", "breakdown": "support breakdown",
               "rejection": "resistance rejection", "breakout": "resistance breakout"}


@dataclass
class Candle:
    start: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Setup:
    kind: str                       # "support" / "resistance"
    level: float
    armed_at: float
    source: str = ""
    touched_at: float | None = None
    extreme: float | None = None    # deepest price through the level since the touch
    checks: int = 0
    prev_close: float | None = None
    off: set = field(default_factory=set)            # branches switched off
    walls: dict = field(default_factory=dict)        # defending walls seen: price -> last record
    fate: str = "none"              # held / consumed / pulled / none

    @property
    def defend(self) -> int:        # +1: support (price above), -1: resistance (price below)
        return 1 if self.kind == "support" else -1


@dataclass
class LevelTest:
    kind: str
    level: float
    source: str
    touched_at: float
    touch_price: float
    context: dict
    mfe_beyond: float = 0.0         # % price went THROUGH the level
    mfe_back: float = 0.0           # % price moved back AWAY on the original side
    outcome: str = "unresolved"     # broke / held / unresolved
    resolved_at: float | None = None
    candidates: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    fate: str = "none"


@dataclass
class Book:
    """Results of one strategy (A or B) for one symbol."""
    equity: float
    closed: list[dict] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)


class SignalEngine:
    def __init__(self, symbol: str, asset: str, levels_fn: Callable[[float], list[float]],
                 rules: Rules | None = None, on_event: Callable[[str, dict], None] | None = None,
                 level_info: Callable[[float], str] | None = None):
        self.symbol, self.asset = symbol, asset
        self.levels_fn = levels_fn
        self.level_info = level_info or (lambda p: "")
        self.rules = rules or Rules()
        self.on_event = on_event or (lambda kind, data: None)
        self.win: deque[dict] = deque()            # last ~2 min of rows
        self.c5: Candle | None = None
        self.c30: Candle | None = None
        self.minute: int | None = None
        self.minute_delta = 0.0
        self.minutes: deque[float] = deque(maxlen=15)   # completed-minute CVD changes, newest last
        self.cvd = 0.0                              # since 00:00 UTC
        self.cvd_day: int | None = None
        self.setups: dict[str, Setup] = {}
        self.recent_levels: dict[tuple[str, float], float] = {}
        self.positions: list[Position] = []
        self.books = {"A": Book(self.rules.start_equity), "B": Book(self.rules.start_equity)}
        self.evaluations: deque[dict] = deque(maxlen=20)
        self.tests: list[LevelTest] = []
        self.finished_tests: deque[dict] = deque(maxlen=10)
        self.last_row: dict | None = None
        self.levels: list[float] = []

    # ------------------------------------------------------------ helpers --
    def _emit(self, kind: str, data: dict) -> None:
        self.on_event(kind, {"symbol": self.symbol, **data})

    def _avg(self, key: str, now: float, window: float) -> tuple[float | None, float]:
        vals = [r[key] for r in self.win if r["t"] > now - window and r.get(key) is not None]
        return (sum(vals) / len(vals) if vals else None), len(vals) / window

    def flow_share(self, now: float) -> tuple[float | None, float]:
        rows = [r for r in self.win if r["t"] > now - self.rules.flow_window]
        buy = sum(r["buy"] for r in rows)
        sell = sum(r["sell"] for r in rows)
        return ((buy / (buy + sell)) if buy + sell > 0 else None), buy + sell

    def cvd_change(self, minutes: int) -> float | None:
        """CVD change over the last n completed minutes (None if not all observed)."""
        if len(self.minutes) < minutes or any(m is None for m in list(self.minutes)[-minutes:]):
            return None
        return sum(list(self.minutes)[-minutes:])

    @staticmethod
    def _roll(c: Candle | None, t: float, tf: int, price: float, hi: float, lo: float):
        start = int(t // tf) * tf
        if c is None:
            return Candle(start, price, hi, lo, price), None
        if start != c.start:
            return Candle(start, price, hi, lo, price), c
        c.high, c.low, c.close = max(c.high, hi), min(c.low, lo), price
        return c, None

    # --------------------------------------------------------------- input --
    def on_row(self, row: dict) -> None:
        t = row["t"]
        if not row.get("valid") or row.get("mid") is None:
            for su in list(self.setups.values()):
                self._log_eval(su, t, "uncertain", None, {"data": False}, {},
                               "data invalid (book out of sync or feed stale)")
            self.setups.clear()
            self.win.clear()
            self.minutes.append(None)          # type: ignore[arg-type]
            self.minute = None
            return
        self.last_row = row
        price = row["last"]
        self.win.append(row)
        while self.win and self.win[0]["t"] <= t - 120:
            self.win.popleft()
        day, minute = int(t // 86400), int(t // 60)
        if day != self.cvd_day:
            self.cvd_day, self.cvd = day, 0.0
        self.cvd += row["buy"] - row["sell"]
        if minute != self.minute:
            if self.minute is not None:
                for _ in range(min(minute - self.minute - 1, 15)):
                    self.minutes.append(None)  # type: ignore[arg-type]   # gap: unknown minutes
                self.minutes.append(self.minute_delta)
            self.minute, self.minute_delta = minute, 0.0
        self.minute_delta += row["buy"] - row["sell"]
        self.c5, done5 = self._roll(self.c5, t, self.rules.confirm_tf, price, row["hi"], row["lo"])
        self.c30, done30 = self._roll(self.c30, t, self.rules.invalidate_tf, price, row["hi"], row["lo"])
        self._update_positions(row)
        self._update_tests(row)
        self.levels = sorted(self.levels_fn(t))
        self._arm(row)
        for su in list(self.setups.values()):
            self._advance(su, row, done5, done30)

    # -------------------------------------------------------------- setups --
    def _arm(self, row: dict) -> None:
        mid, t, r = row["mid"], row["t"], self.rules
        for kind in ("support", "resistance"):
            if kind in self.setups:
                continue
            cands = [L for L in self.levels if abs(mid - L) / L <= r.near_pct
                     and (L <= mid if kind == "support" else L >= mid)]
            if not cands:
                continue
            L = min(cands, key=lambda x: abs(x - mid))
            if any(su.level == L for su in self.setups.values()):
                continue
            used = self.recent_levels.get((kind, round(L, 8)))
            if used is not None and t - used < 7200:
                continue
            self.setups[kind] = Setup(kind, L, t, source=self.level_info(L))

    def _track_walls(self, su: Setup, row: dict) -> None:
        """Follow the large orders defending the level: still there, traded away, or cancelled."""
        r = self.rules
        side = "walls_bid" if su.kind == "support" else "walls_ask"
        wmin = r.wall_min_for(self.asset)
        now = {round(w["price"], 8): w for w in (row.get(side) or [])
               if abs(w["price"] - su.level) / su.level <= r.wall_near_pct and w["peak"] >= wmin}
        for p, w in now.items():
            su.walls[p] = w
        for p in [p for p in su.walls if p not in now]:
            w = su.walls.pop(p)
            gone = w["peak"]
            su.fate = "consumed" if w.get("hit", 0.0) >= 0.5 * gone else "pulled"
        if now:
            big = max(now.values(), key=lambda w: w["coins"])
            su.fate = "absorbing" if big.get("hit", 0.0) >= 0.25 * big["peak"] else "held"

    def _advance(self, su: Setup, row: dict, done5: Candle | None, done30: Candle | None) -> None:
        r, t, L, d = self.rules, row["t"], su.level, su.defend
        self._track_walls(su, row)
        if su.touched_at is None:
            if (row["lo"] <= L * (1 + r.touch_pct)) if d > 0 else (row["hi"] >= L * (1 - r.touch_pct)):
                su.touched_at = t
                su.extreme = row["lo"] if d > 0 else row["hi"]
                self._start_test(su, row)
            elif abs(row["mid"] - L) / L > 2 * r.near_pct:
                del self.setups[su.kind]
            return
        su.extreme = min(su.extreme, row["lo"]) if d > 0 else max(su.extreme, row["hi"])
        if done30 is not None:
            through = d * (done30.close - L) < 0
            rev, brk = BRANCHES[su.kind][0][0], BRANCHES[su.kind][1][0]
            if through and rev not in su.off:
                su.off.add(rev)
                self._log_eval(su, t, rev, None, {"price": False}, {},
                               f"{rev} switched off: 30-min candle closed through the level")
            elif not through and brk not in su.off and su.prev_close is not None and d * (su.prev_close - L) < 0:
                su.off.add(brk)
                self._log_eval(su, t, brk, None, {"price": False}, {},
                               f"{brk} switched off: 30-min candle closed back on the original side")
            if len(su.off) == 2:
                self._drop(su, t)
                return
        if done5 is None or done5.start + r.confirm_tf <= su.touched_at:
            return
        su.checks += 1
        self._decide(su, row, done5)
        su.prev_close = done5.close
        if su.kind in self.setups and su.checks >= r.give_up_candles:
            self._log_eval(su, t, "uncertain", None, {}, {},
                           f"expired: no signal within {r.give_up_candles} five-minute closes")
            self._drop(su, t)

    def _drop(self, su: Setup, t: float) -> None:
        if self.setups.get(su.kind) is su:
            del self.setups[su.kind]
        self.recent_levels[(su.kind, round(su.level, 8))] = t

    # ------------------------------------------------------------- decision --
    def candidate(self, su: Setup, close: float, prev_close: float | None) -> tuple[str, str] | None:
        """(branch, trade side) suggested by this 5-min close, or None = uncertain."""
        L, d, b = su.level, su.defend, self.rules.break_buf
        rev, brk = BRANCHES[su.kind]
        if d * (close - L) > 0:
            return rev
        if (d * (L * (1 - d * b) - close) > 0 and prev_close is not None and d * (prev_close - L) < 0):
            return brk
        return None

    def conditions(self, su: Setup, branch: str, trade: str, row: dict, close: float | None,
                   prev_close: float | None) -> tuple[dict, dict]:
        """Checklist for one branch. Returns (passed flags, measured values)."""
        r, t, L = self.rules, row["t"], su.level
        s = 1 if trade == "long" else -1
        reversal = branch in ("bounce", "rejection")
        walls = list(su.walls.values())
        wmin = r.wall_min_for(self.asset)

        def defending(w):
            shrunk = w["peak"] - w["coins"]
            by_trades = w.get("hit", 0.0) >= 0.5 * shrunk if shrunk > 0 else True
            return (w["age"] >= r.wall_min_age and w["coins"] >= wmin
                    and (w["coins"] >= (1 - r.wall_max_shrink) * w["peak"] or by_trades))
        good = [w for w in walls if defending(w)]
        best = max(walls, key=lambda w: w["coins"]) if walls else None
        i10, c10 = self._avg("imb10", t, r.imb_window)
        i25, c25 = self._avg("imb25", t, r.imb_window)
        share, _ = self.flow_share(t)
        side_share = None if share is None else (share if s > 0 else 1 - share)
        cvd1, cvd5, cvd15 = self.cvd_change(1), self.cvd_change(5), self.cvd_change(15)
        cvd_used = cvd1 if reversal else cvd5
        cand = self.candidate(su, close, prev_close) if close is not None else None
        ok = {
            "wall": bool(good) if reversal else not good,
            "imbalance": (i10 is not None and i25 is not None and min(c10, c25) >= r.imb_coverage
                          and s * i10 >= r.imb_min and s * i25 >= r.imb_min),
            "flow": side_share is not None and side_share >= r.flow_share,
            "cvd": cvd_used is not None and s * cvd_used > 0,
            "price": cand is not None and cand[0] == branch,
        }
        vals = {"wall_coins": None if best is None else best["coins"],
                "wall_price": None if best is None else best["price"],
                "wall_age": None if best is None else best["age"],
                "wall_hit": None if best is None else best.get("hit", 0.0),
                "wall_fate": su.fate,
                "imb10": i10, "imb25": i25, "imb_coverage": min(c10, c25),
                "flow_share": side_share, "cvd_1m": cvd1, "cvd_5m": cvd5, "cvd_15m": cvd15,
                "candle_close": close, "prev_close": prev_close}
        return ok, vals

    def _decide(self, su: Setup, row: dict, candle: Candle) -> None:
        r, t, L = self.rules, row["t"], su.level
        cand = self.candidate(su, candle.close, su.prev_close)
        test = next((x for x in self.tests if x.level == L and x.touched_at == su.touched_at), None)
        if cand is None or cand[0] in su.off:
            label = "uncertain" if cand is None else cand[0]
            why = ("uncertain: the 5-min close is neither back on the defended side nor a confirmed break"
                   if cand is None else f"{cand[0]} candidate, but it was switched off earlier")
            ok, vals = self.conditions(su, BRANCHES[su.kind][0][0], BRANCHES[su.kind][0][1], row,
                                       candle.close, su.prev_close)
            if test is not None:
                test.candidates.append("uncertain")
            self._log_eval(su, t, label, vals, {"price": False}, {}, why)
            return
        branch, trade = cand
        if test is not None:
            test.candidates.append(branch)
        s = 1 if trade == "long" else -1
        ok, vals = self.conditions(su, branch, trade, row, candle.close, su.prev_close)
        # risk: reversal stops go beyond the deepest push through the level; break stops on the other side of it
        if branch in ("bounce", "rejection"):
            stop = (min(L, su.extreme) * (1 - r.stop_buf)) if s > 0 else (max(L, su.extreme) * (1 + r.stop_buf))
        else:
            stop = L * (1 - r.stop_buf) if s > 0 else L * (1 + r.stop_buf)
        entry_a = row["ask"] * (1 + r.slip) if s > 0 else row["bid"] * (1 - r.slip)
        entry_b = min(L * (1 + r.retest_pct), row["bid"]) if s > 0 else max(L * (1 - r.retest_pct), row["ask"])
        nxt = ([x for x in self.levels if x > entry_a * (1 + r.near_pct)] if s > 0
               else [x for x in self.levels if x < entry_a * (1 - r.near_pct)])
        target = None
        if nxt:
            target = (min(nxt) * (1 - r.target_buf)) if s > 0 else (max(nxt) * (1 + r.target_buf))
        rr_a = plan_rr(trade, entry_a, stop, target, r.taker, r) if target else None
        rr_b = plan_rr(trade, entry_b, stop, target, r.maker, r) if target else None
        ok["risk"] = (rr_a is not None and rr_a >= r.rr_min) or (rr_b is not None and rr_b >= r.rr_min)
        vals.update({"stop": stop, "target": target, "entry_a": entry_a, "entry_b": entry_b,
                     "rr_a": rr_a, "rr_b": rr_b, "trade": trade})
        busy = any(p.status in ("pending", "open") for p in self.positions)
        setup_ok = all(ok.values())
        if setup_ok and busy:
            reason = "all conditions met, but a trade is already open on this coin"
        elif setup_ok:
            reason = "SIGNAL"
        else:
            reason = f"{branch} candidate: " + ", ".join(k for k, v in ok.items() if not v) + " not met"
        self._log_eval(su, t, branch, vals, ok, {}, reason)
        if not setup_ok or busy:
            return
        name = SETUP_NAMES[branch]
        self._drop(su, t)
        self._emit("signal", {"time": t, "side": trade, "setup": name, "level": L, **vals})
        for strat, entry, rr, fee in (("A", entry_a, rr_a, r.taker), ("B", entry_b, rr_b, r.maker)):
            if rr is None or rr < r.rr_min:
                self._emit("skip", {"time": t, "strategy": strat, "side": trade, "setup": name,
                                    "why": f"reward:risk {0 if rr is None else rr:.2f} < {r.rr_min}"})
                continue
            p = Position(strat, self.symbol, trade, L, entry, stop, target, t, setup=name)
            if strat == "A":
                open_position(p, self.books["A"].equity, r, t, fee)
            self.positions.append(p)
            if test is not None:
                test.trades.append(f"{strat} {name}")
            self._emit("open" if strat == "A" else "limit", {"time": t, **p.as_dict()})

    def _log_eval(self, su: Setup, t: float, candidate: str, vals: dict | None, ok: dict, _x: dict,
                  reason: str) -> None:
        ev = {"time": t, "symbol": self.symbol, "kind": su.kind, "source": su.source, "level": su.level,
              "candidate": candidate, "side": (vals or {}).get("trade") or dict(BRANCHES[su.kind]).get(candidate, ""),
              "passed": reason == "SIGNAL", "reason": reason,
              **{f"ok_{k}": v for k, v in ok.items()}, **(vals or {})}
        self.evaluations.appendleft(ev)
        self._emit("evaluation", ev)

    # ---------------------------------------------------------- level tests --
    def _start_test(self, su: Setup, row: dict) -> None:
        t = row["t"]
        share, _ = self.flow_share(t)
        i10, _ = self._avg("imb10", t, self.rules.imb_window)
        i25, _ = self._avg("imb25", t, self.rules.imb_window)
        ctx = {"cvd_1m": self.cvd_change(1), "cvd_5m": self.cvd_change(5), "cvd_15m": self.cvd_change(15),
               "buy_share": share, "imb10": i10, "imb25": i25,
               "wall_coins": max((w["coins"] for w in su.walls.values()), default=None)}
        self.tests.append(LevelTest(su.kind, su.level, su.source, t, row["last"], ctx))

    def _update_tests(self, row: dict) -> None:
        r, t = self.rules, row["t"]
        for x in list(self.tests):
            d = 1 if x.kind == "support" else -1
            beyond = d * (x.level - (row["lo"] if d > 0 else row["hi"])) / x.level * 100
            back = d * ((row["hi"] if d > 0 else row["lo"]) - x.level) / x.level * 100
            x.mfe_beyond = max(x.mfe_beyond, beyond)
            x.mfe_back = max(x.mfe_back, back)
            su = self.setups.get(x.kind)
            if su is not None and su.level == x.level:
                x.fate = su.fate
            if x.outcome == "unresolved" and x.resolved_at is None:
                if beyond >= r.outcome_pct * 100:
                    x.outcome, x.resolved_at = "broke", t
                elif back >= r.outcome_pct * 100:
                    x.outcome, x.resolved_at = "held", t
            if t - x.touched_at >= r.outcome_horizon:
                self.tests.remove(x)
                rec = {"time": x.touched_at, "symbol": self.symbol, "kind": x.kind, "level": x.level,
                       "source": x.source, "touch_price": x.touch_price, "outcome": x.outcome,
                       "minutes_to_outcome": None if x.resolved_at is None else round((x.resolved_at - x.touched_at) / 60, 1),
                       "max_beyond_pct": round(x.mfe_beyond, 3), "max_back_pct": round(x.mfe_back, 3),
                       "wall_fate": x.fate, "candidates": " ".join(x.candidates), "trades": " ".join(x.trades),
                       **{f"at_touch_{k}": v for k, v in x.context.items()}}
                self.finished_tests.appendleft(rec)
                self._emit("level_test", rec)

    # ----------------------------------------------------------- positions --
    def _update_positions(self, row: dict) -> None:
        for p in list(self.positions):
            book = self.books[p.strategy]
            ev = update_position(p, row, self.rules, book.equity)
            if ev is None:
                continue
            if ev in ("fill", "fill+close"):
                self._emit("fill", {"time": row["t"], **p.as_dict()})
            if p.status == "missed":
                book.missed.append(p.as_dict())
                self.positions.remove(p)
                self._emit("missed", {"time": row["t"], **p.as_dict()})
            elif p.status == "closed":
                book.equity += p.pnl
                book.closed.append(p.as_dict())
                self.positions.remove(p)
                self._emit("close", {"time": row["t"], **p.as_dict()})

    # ---------------------------------------------------------------- view --
    def stats(self) -> dict:
        out = {k: {**summarize(b.closed, len(b.missed)), "equity": b.equity} for k, b in self.books.items()}
        by: dict[str, dict] = {}
        for name in SETUP_NAMES.values():
            row = {}
            for k, b in self.books.items():
                closed = [x for x in b.closed if x.get("setup") == name]
                missed = [x for x in b.missed if x.get("setup") == name]
                row[k] = summarize(closed, len(missed))
            by[name] = row
        out["by_setup"] = by
        return out

    def view(self) -> dict:
        row = self.last_row
        out = {"symbol": self.symbol, "stats": self.stats(),
               "positions": [p.as_dict() for p in self.positions],
               "evaluations": list(self.evaluations)[:8], "level_tests": list(self.finished_tests)[:5],
               "cvd": self.cvd, "setups": [], "nearest": None}
        if row is None:
            return out
        mid = row["mid"]
        below = [x for x in self.levels if x < mid]
        above = [x for x in self.levels if x > mid]
        out["nearest"] = {"support": below[-1] if below else None, "resistance": above[0] if above else None,
                          "mid": mid}
        for kind in ("support", "resistance"):
            su = self.setups.get(kind)
            if su is None:
                continue
            close = self.c5.close if self.c5 else None
            cand = self.candidate(su, close, su.prev_close) if close is not None else None
            branch, trade = cand if cand else BRANCHES[kind][0]
            ok, vals = self.conditions(su, branch, trade, row, close, su.prev_close)
            out["setups"].append({"kind": kind, "level": su.level, "source": su.source,
                                  "touched": su.touched_at is not None, "checks": su.checks,
                                  "leaning": cand[0] if cand else "uncertain", "branch": branch, "trade": trade,
                                  "off": sorted(su.off), "fate": su.fate, "ok": ok, "values": vals})
        share, _ = self.flow_share(row["t"])
        i10, _ = self._avg("imb10", row["t"], self.rules.imb_window)
        i25, _ = self._avg("imb25", row["t"], self.rules.imb_window)
        out["live"] = {"imb10": i10, "imb25": i25, "buy_share": share, "cvd_1m": self.cvd_change(1),
                       "cvd_5m": self.cvd_change(5), "cvd_15m": self.cvd_change(15)}
        return out
