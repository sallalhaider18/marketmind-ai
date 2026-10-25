"""S/R Order-Flow Analyzer: YOUR support/resistance levels, judged by the order book only.

You add the levels you use (dashboard or API). Each second, for every level
within 1% of price, it measures the defending depth and what executed trades
do to it (see measure.py) and gives a verdict: BREAKOUT / REVERSAL pressure
(HIGH or MODERATE) or INCONCLUSIVE. With no levels of your own, it watches
the strong daily/4H levels instead (marked "auto").

Every VISIT (price within 0.1% of a level) is recorded:
  * the first non-inconclusive verdict during the visit (before the outcome),
  * the outcome, decided without candles: price moved 0.3% THROUGH the level
    first (broke), 0.3% BACK first (held), or neither within 60 min (unresolved),
  * the per-second measurements, saved so the verdict rules can be re-scored.
Files: reports/sr_visits.csv, data/sr_analyzer/mexc_futures/<SYMBOL>/<date>/*.parquet
Your levels: reports/my_levels.json
"""
from __future__ import annotations

import csv
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from app.sranalyzer.measure import (AnalyzerRules, classify, defend_side, readings, second_record, verdict,
                                    window, zone)
from app.sranalyzer.scoring import FailedCross, ImpactBaseline, ScoreRules, features, scores, verdict2

log = logging.getLogger(__name__)

VISIT_FIELDS = ["time", "symbol", "level", "kind", "origin", "call", "call_strength", "call_time",
                "call_reason", "call2", "call2_time", "call2_reason", "breakout_score_at_call",
                "reversal_score_at_call", "max_breakout_score", "max_reversal_score", "failed_cross",
                "overlap", "outcome", "resolved_time", "seconds_to_call", "max_through_pct", "max_back_pct",
                "raw_file"]
STRONG2 = ("breakout", "reversal")


def _iso(t: float | None) -> str:
    return "" if t is None else datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class Visit:
    kind: str
    start: float
    origin: str
    call: str = "inconclusive"
    strength: str = ""
    call_time: float | None = None
    call_reason: str = ""
    outcome: str = "unresolved"
    resolved: float | None = None
    through: float = 0.0
    back: float = 0.0
    raw: list = field(default_factory=list)
    call2: str = "inconclusive"         # scored version (v2): first STRONG verdict before the outcome
    call2_time: float | None = None
    call2_reason: str = ""
    b_at_call: float | None = None
    r_at_call: float | None = None
    max_b: float = 0.0
    max_r: float = 0.0
    failed_cross: bool = False
    overlap: bool = False               # another visit on this coin was running: not independent


@dataclass
class Watch:
    level: float
    kind: str
    prev_zone: dict | None = None
    records: deque = field(default_factory=lambda: deque(maxlen=90))
    visit: Visit | None = None
    view: dict | None = None
    cross: FailedCross = field(default_factory=FailedCross)


class LevelStore:
    """Your levels per symbol, saved to a small JSON file."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data: dict[str, list[float]] = json.loads(path.read_text())
        except Exception:
            self.data = {}

    def get(self, symbol: str) -> list[float]:
        return sorted(self.data.get(symbol, []))

    def add(self, symbol: str, price: float) -> list[float]:
        cur = self.data.setdefault(symbol, [])
        if not any(abs(p - price) / price < 1e-4 for p in cur):
            cur.append(float(price))
        self._save()
        return self.get(symbol)

    def remove(self, symbol: str, price: float) -> list[float]:
        self.data[symbol] = [p for p in self.data.get(symbol, []) if abs(p - price) / price >= 1e-4]
        self._save()
        return self.get(symbol)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1))
        except Exception:
            log.exception("could not save %s", self.path)


class SRAnalyzer:
    def __init__(self, symbols: list[str], store: LevelStore, reports: Path, raw_dir: Path | None,
                 auto_levels: Callable[[str], list[float]] | None = None, rules: AnalyzerRules | None = None,
                 out: Callable[[str], None] = print, score_rules: ScoreRules | None = None):
        self.symbols = symbols
        self.store = store
        self.reports = reports
        self.raw_dir = raw_dir
        self.auto_levels = auto_levels or (lambda s: [])
        self.rules = rules or AnalyzerRules()
        self.sr = score_rules or ScoreRules()
        self.say = out
        self.impact = {s: ImpactBaseline() for s in symbols}
        self.last_trade: dict[str, float | None] = {s: None for s in symbols}   # local time of last trade
        self.lags: dict[str, deque] = {}
        self.clock_offset: dict[str, float | None] = {s: None for s in symbols}
        self.last_resync: dict[str, tuple[int, float]] = {}
        self.watches: dict[str, dict[float, Watch]] = {s: {} for s in symbols}
        self.visits_done: dict[str, deque] = {s: deque(maxlen=8) for s in symbols}
        self.scores: dict[str, list[dict]] = {s: [] for s in symbols}
        self.mid: dict[str, float | None] = {s: None for s in symbols}
        self._load_history()

    # ------------------------------------------------------------- levels --
    def levels_for(self, symbol: str) -> tuple[list[float], str]:
        mine = self.store.get(symbol)
        if mine:
            return mine, "mine"
        return sorted(self.auto_levels(symbol)), "auto"

    # -------------------------------------------------------------- input --
    def _quality(self, symbol: str, t: float, trades: list, health: dict | None) -> tuple[bool, str]:
        """Feed health for this second: book resynced after a gap, trades later than usual,
        trade stream gone quiet.

        Trade times are MEXC's clock, `t` is this computer's clock. They can differ by a few
        seconds (a PC clock that is off), so "late" means later than the usual difference:
        the smallest delay seen in the last 10 minutes is taken as the clock offset."""
        sr = self.sr
        if health:
            # only real resyncs (a gap or a broken book). The routine refresh every few minutes
            # rebuilds the book without missing anything, so it does not count.
            n = health.get("resyncs") or 0
            prev = self.last_resync.get(symbol)
            if prev is None or n != prev[0]:
                self.last_resync[symbol] = (n, t if prev is not None else -1e18)
            if t - self.last_resync[symbol][1] < sr.resync_quiet:
                return False, "order book had to be resynced in the last minute"
        if trades:
            self.last_trade[symbol] = t                       # local arrival time
            lag = t - max(x.timestamp.timestamp() for x in trades)
            lags = self.lags.setdefault(symbol, deque())
            lags.append((t, lag))
            while lags and lags[0][0] < t - 600:
                lags.popleft()
            offset = min(x[1] for x in lags)
            self.clock_offset[symbol] = offset
            if lag - offset > sr.max_trade_lag:
                return False, f"trades arriving {lag - offset:.0f} s later than usual"
        last = self.last_trade[symbol]
        if last is None or t - last > sr.max_trade_gap:
            return False, "trade stream quiet: no trades for 30 s"
        return True, ""

    def on_tick(self, symbol: str, t: float, book, trades: list, cs: float | None,
                health: dict | None = None) -> None:
        watches = self.watches[symbol]
        if book is None or cs is None or book.mid is None:
            for w in watches.values():          # blind: forget the per-second history
                w.prev_zone = None
                w.records.clear()
            return
        mid = book.mid
        self.mid[symbol] = mid
        r = self.rules
        q_ok, q_reason = self._quality(symbol, t, trades, health)
        buys = sum(x.contracts for x in trades if x.side == "buy") * cs
        sells = sum(x.contracts for x in trades if x.side != "buy") * cs
        self.impact[symbol].add(t, mid, buys, sells)
        base = self.impact[symbol].value()
        last_px = trades[-1].price if trades else mid
        asset = symbol.replace("USDT.P", "").replace("USDT", "")
        levels, origin = self.levels_for(symbol)
        for L in levels:
            if abs(mid - L) / L <= r.watch_pct and L not in watches:
                watches[L] = Watch(L, "resistance" if L > mid else "support")
        for L, w in list(watches.items()):
            near = abs(mid - L) / L <= r.watch_pct
            if L not in levels and w.visit is None:
                del watches[L]
                continue
            if not near and w.visit is None:
                del watches[L]
                continue
            if w.visit is None:
                kind = "resistance" if L > mid else "support"
                if kind != w.kind:                # price crossed it while not visiting: start over
                    w.kind, w.prev_zone = kind, None
                    w.records.clear()
            side = book.asks if defend_side(w.kind) == "asks" else book.bids
            z = zone(side, L, r.wide_pct)
            rec = second_record(t, mid, w.kind, L, w.prev_zone, z, trades, cs, r)
            rec.update({"last": last_px, "zone": {p: q * cs for p, q in z.items()}, "impact_base": base,
                        "q_ok": q_ok, "q_reason": q_reason})
            w.prev_zone = z
            w.records.append(rec)
            recs = list(w.records)
            w15, w60 = window(recs, t, 15, r), window(recs, t, 60, r)
            w5 = window(recs, t, 5, r)
            rec["share15"] = w15["attack_share"] if w15 else None
            c15, why15 = classify(w15, w.kind, L, r)
            c60, why60 = classify(w60, w.kind, L, r)
            label, strength = verdict(c15, c60)
            # scored version (v2)
            if w.cross.update(t, last_px, w.kind, L, rec["share15"], self.sr) and w.visit is not None:
                w.visit.failed_cross = True
            f = features(recs, w15, w60, w.kind, L, asset, w.cross.detected_at, t, self.sr)
            b, rv = scores(f, self.sr)
            label2, why2 = verdict2(f, b, rv, self.sr)
            w.view = {"level": L, "kind": w.kind, "origin": origin if L in levels else "old",
                      "distance_pct": (L / mid - 1) * 100, "verdict": label, "strength": strength,
                      "reason": why60, "windows": {k: v for k, v in (("5s", w5), ("15s", w15), ("60s", w60))},
                      "classes": {"15s": c15, "60s": c60}, "readings": readings(w60, w.kind, L, r),
                      "depth10": rec["depth10"], "depth25": rec["depth25"],
                      "visiting": w.visit is not None,
                      "v2": {"verdict": label2, "reason": why2, "breakout_score": b, "reversal_score": rv,
                             "quality_ok": f.get("quality_ok"),
                             **{k: f.get(k) for k in ("impact_ratio", "survival", "persistence", "failed_cross",
                                                      "progress_pct", "attack_share")}}}
            self._visit(symbol, w, rec, label, strength, why60, origin, (label2, why2, b, rv))

    # ------------------------------------------------------------- visits --
    def _visit(self, symbol: str, w: Watch, rec: dict, label: str, strength: str, why: str, origin: str,
               v2: tuple) -> None:
        r, t, mid, L = self.rules, rec["t"], rec["mid"], w.level
        d = 1 if w.kind == "resistance" else -1
        if w.visit is None:
            if abs(mid - L) / L <= r.visit_pct:
                busy = any(o.visit is not None for o in self.watches[symbol].values() if o is not w)
                w.visit = Visit(w.kind, t, origin, overlap=busy)
            else:
                return
        v = w.visit
        v.raw.append(rec)
        label2, why2, b, rv = v2
        v.max_b, v.max_r = max(v.max_b, b or 0.0), max(v.max_r, rv or 0.0)
        if v.resolved is None and label2 in STRONG2 and v.call2_time is None:
            v.call2, v.call2_time, v.call2_reason, v.b_at_call, v.r_at_call = label2, t, why2, b, rv
            go = "LONG" if (w.kind == "resistance") == (label2 == "breakout") else "SHORT"
            self.say(f"[S/R v2] {symbol} {w.kind} {L:,.2f}: {go} - {label2.upper()} "
                     f"(breakout {b:.0f} / reversal {rv:.0f}) - {why2}")
        through = d * (mid - L) / L * 100
        v.through, v.back = max(v.through, through), max(v.back, -through)
        if v.resolved is None:
            if label != "inconclusive" and v.call_time is None:
                v.call, v.strength, v.call_time, v.call_reason = label, strength, t, why
                go = "LONG" if (w.kind == "resistance") == (label == "breakout") else "SHORT"
                kind = "BREAKOUT" if label == "breakout" else "REVERSAL"
                head = f"{go} - {kind} (HIGH)" if strength == "HIGH" else f"GET READY: possible {go} - {kind} (MODERATE)"
                self.say(f"[S/R] {symbol} {w.kind} {L:,.2f}: {head} ({label} pressure {strength}) - {why}")
            if through >= r.outcome_pct * 100:
                v.outcome, v.resolved = "broke", t
            elif -through >= r.outcome_pct * 100:
                v.outcome, v.resolved = "held", t
        if t - v.start >= r.horizon:
            self._finish(symbol, w)

    def _finish(self, symbol: str, w: Watch) -> None:
        v = w.visit
        w.visit = None
        raw_file = ""
        if self.raw_dir is not None and v.raw:
            t0 = datetime.fromtimestamp(v.start, tz=timezone.utc)
            path = (self.raw_dir / symbol / t0.strftime("%Y-%m-%d") /
                    f"{t0.strftime('%H%M%S')}_{w.level:.4f}.parquet")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(v.raw)
                if "zone" in df:
                    df["zone"] = df["zone"].map(lambda z: json.dumps({str(k): v for k, v in (z or {}).items()}))
                df.to_parquet(path, index=False)
                raw_file = str(path)
            except Exception:
                log.exception("could not save visit data")
        row = {"time": _iso(v.start), "symbol": symbol, "level": w.level, "kind": v.kind, "origin": v.origin,
               "call": v.call, "call_strength": v.strength, "call_time": _iso(v.call_time),
               "call_reason": v.call_reason, "call2": v.call2, "call2_time": _iso(v.call2_time),
               "call2_reason": v.call2_reason, "breakout_score_at_call": v.b_at_call,
               "reversal_score_at_call": v.r_at_call, "max_breakout_score": v.max_b,
               "max_reversal_score": v.max_r, "failed_cross": v.failed_cross, "overlap": v.overlap,
               "outcome": v.outcome, "resolved_time": _iso(v.resolved),
               "seconds_to_call": None if v.call_time is None else round(v.call_time - v.start),
               "max_through_pct": round(v.through, 3), "max_back_pct": round(v.back, 3), "raw_file": raw_file}
        self.visits_done[symbol].appendleft(row)
        self.scores[symbol].append(row)
        self.say(f"[S/R VISIT] {symbol} {v.kind} {w.level:,.2f}: baseline called {v.call.upper()} {v.strength}, "
                 f"v2 called {v.call2.upper()}, outcome {v.outcome.upper()}"
                 + (" (overlapping visit: not counted as independent)" if v.overlap else ""))
        path = self.reports / "sr_visits.csv"
        new = not path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=VISIT_FIELDS, extrasaction="ignore")
                if new:
                    wr.writeheader()
                wr.writerow(row)
        except Exception:
            log.exception("could not write sr_visits.csv")

    def _load_history(self) -> None:
        path = self.reports / "sr_visits.csv"
        if not path.exists():
            return
        try:
            df = pd.read_csv(path).fillna("")
        except Exception:
            return
        for row in df.to_dict("records"):
            if row.get("symbol") in self.scores:
                self.scores[row["symbol"]].append(row)
                self.visits_done[row["symbol"]].appendleft(row)

    # --------------------------------------------------------------- view --
    def scorecard(self, symbol: str) -> dict:
        all_rows = self.scores[symbol]
        rows = [x for x in all_rows if str(x.get("overlap", "")).lower() not in ("true", "1")]
        n = len(rows)

        def pct(xs, outcome):
            return (sum(1 for x in xs if x["outcome"] == outcome) / len(xs)) if xs else None
        bo = [x for x in rows if x["call"] == "breakout"]
        rv = [x for x in rows if x["call"] == "reversal"]
        bo2 = [x for x in rows if x.get("call2") == "breakout"]
        rv2 = [x for x in rows if x.get("call2") == "reversal"]
        return {"visits": n, "overlapping_excluded": len(all_rows) - n,
                "unresolved_rate": pct(rows, "unresolved"),
                "broke_rate": pct(rows, "broke"), "held_rate": pct(rows, "held"),
                "breakout_calls": len(bo), "breakout_right": pct(bo, "broke"),
                "reversal_calls": len(rv), "reversal_right": pct(rv, "held"),
                "inconclusive": n - len(bo) - len(rv),
                "v2": {"breakout_calls": len(bo2), "breakout_right": pct(bo2, "broke"),
                       "reversal_calls": len(rv2), "reversal_right": pct(rv2, "held")}}

    def view(self, symbol: str) -> dict:
        levels, origin = self.levels_for(symbol)
        watched = sorted((w.view for w in self.watches[symbol].values() if w.view),
                         key=lambda v: abs(v["distance_pct"]))
        return {"levels": levels, "origin": origin, "mid": self.mid[symbol], "watched": watched,
                "clock_offset_s": self.clock_offset[symbol],
                "recent_visits": list(self.visits_done[symbol])[:5], "scorecard": self.scorecard(symbol),
                "rules": {"watch_pct": self.rules.watch_pct, "visit_pct": self.rules.visit_pct,
                          "outcome_pct": self.rules.outcome_pct}}
