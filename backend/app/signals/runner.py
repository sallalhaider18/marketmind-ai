"""Live wiring for the signal engine: levels, logs, terminal messages, dashboard view.

Logs (reports/):
  signal_evaluations.csv  every check, passed or rejected, with every measured value
  signal_trades.csv       every paper trade of strategy A and B (closed and missed)
Closed results are re-read from signal_trades.csv at start-up, so the
statistics survive a restart. Open paper positions do not.
"""
from __future__ import annotations

import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from app.obtrade.levels import find_levels
from app.signals.engine import SignalEngine
from app.signals.rules import Rules
from app.signals.rows import RowRecorder

log = logging.getLogger(__name__)

EVAL_FIELDS = ["time", "symbol", "kind", "source", "level", "candidate", "side", "passed", "reason", "ok_data",
               "ok_wall", "ok_imbalance", "ok_flow", "ok_cvd", "ok_price", "ok_risk", "wall_coins", "wall_price",
               "wall_age", "wall_hit", "wall_fate", "imb10", "imb25", "imb_coverage", "flow_share", "cvd_1m",
               "cvd_5m", "cvd_15m", "candle_close", "prev_close", "stop", "target", "entry_a", "entry_b",
               "rr_a", "rr_b"]
TRADE_FIELDS = ["strategy", "symbol", "setup", "side", "level", "entry", "stop", "target", "signal_time", "status",
                "qty", "fill_time", "exit_time", "exit_price", "result", "fees", "pnl", "r", "risk_usd"]
TEST_FIELDS = ["time", "symbol", "kind", "level", "source", "touch_price", "outcome", "minutes_to_outcome",
               "max_beyond_pct", "max_back_pct", "wall_fate", "candidates", "trades", "at_touch_cvd_1m",
               "at_touch_cvd_5m", "at_touch_cvd_15m", "at_touch_buy_share", "at_touch_imb10", "at_touch_imb25",
               "at_touch_wall_coins"]


def level_source(lv) -> str:
    """Which charts made this level: 1D, 4H or both."""
    if lv.daily and lv.touches > lv.daily:
        return "1D+4H"
    return "1D" if lv.daily else "4H"


def _iso(t: float | None) -> str:
    return "" if t is None else datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")


class SignalRunner:
    def __init__(self, symbols: list[str], candles: Callable[[str, str], pd.DataFrame], reports: Path,
                 recorder: RowRecorder | None = None, rules: Rules | None = None,
                 out: Callable[[str], None] = print):
        self.symbols = symbols
        self.candles = candles
        self.reports = reports
        self.recorder = recorder
        self.rules = rules or Rules()
        self.say = out
        self.levels: dict[str, list[float]] = {s: [] for s in symbols}
        self.sources: dict[str, dict[float, str]] = {s: {} for s in symbols}
        self.engines = {s: SignalEngine(s, s.replace("USDT.P", "").replace("USDT", ""),
                                        levels_fn=(lambda t, s=s: self.levels[s]),
                                        rules=self.rules, on_event=self._event,
                                        level_info=(lambda p, s=s: self.sources[s].get(p, ""))) for s in symbols}
        self._restore()

    # ------------------------------------------------------------- inputs --
    def on_row(self, symbol: str, row: dict) -> None:
        if self.recorder:
            self.recorder.add(symbol, row)
        self.engines[symbol].on_row(row)

    async def run_levels(self, mid: Callable[[str], float | None], every: float = 1800) -> None:
        while True:
            for s in self.symbols:
                try:
                    df4 = await asyncio.to_thread(self.candles, s, "4h")
                    df1 = await asyncio.to_thread(self.candles, s, "1d")
                    found = find_levels(df4, df1, current=mid(s))
                    self.levels[s] = [lv.price for lv in found]
                    self.sources[s] = {lv.price: level_source(lv) for lv in found}
                except Exception as exc:
                    log.warning("signal levels for %s failed: %s", s, exc)
            await asyncio.sleep(every if all(self.levels.values()) else 60)

    async def run_recorder(self, every: float = 30) -> None:
        if not self.recorder:
            return
        import time
        try:
            while True:
                for s in self.recorder.due(time.time()):
                    await asyncio.to_thread(self.recorder.flush, s)
                await asyncio.sleep(every)
        finally:
            self.recorder.flush_all()

    # ------------------------------------------------------------- output --
    def _append(self, name: str, fields: list[str], row: dict) -> None:
        path = self.reports / name
        new = not path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception:
            log.exception("could not write %s", name)

    def _event(self, kind: str, d: dict) -> None:
        sym = d["symbol"]
        if kind == "evaluation":
            self._append("signal_evaluations.csv", EVAL_FIELDS, {**d, "time": _iso(d["time"])})
            return
        if kind == "level_test":
            self._append("level_tests.csv", TEST_FIELDS, {**d, "time": _iso(d["time"])})
            self.say(f"[LEVEL TEST] {sym} {d['kind']} {d['level']:,.2f}: {d['outcome'].upper()} "
                     f"(went {d['max_beyond_pct']:.2f}% through, {d['max_back_pct']:.2f}% back; wall {d['wall_fate']})")
            return
        if kind == "signal":
            self.say(f"[SIGNAL] {sym} {d['setup'].upper()} -> {d['side'].upper()} at level {d['level']:,.2f}: all checks passed. "
                     f"Stop {d['stop']:,.2f}, target {d['target']:,.2f}, "
                     f"R:R A {d['rr_a'] or 0:.2f} / B {d['rr_b'] or 0:.2f}")
        elif kind == "open":
            self.say(f"[A] {sym} {d['side']} entered at {d['entry']:,.2f} (market)")
        elif kind == "limit":
            self.say(f"[B] {sym} {d['side']} limit placed at {d['entry']:,.2f} (retest)")
        elif kind == "fill":
            self.say(f"[B] {sym} limit filled at {d['entry']:,.2f}")
        elif kind == "skip":
            self.say(f"[{d['strategy']}] {sym} skipped: {d['why']}")
        elif kind in ("close", "missed"):
            if kind == "close":
                self.say(f"[{d['strategy']}] {sym} {d['side']} closed at {d['exit_price']:,.2f} "
                         f"({d['result']}): {d['r']:+.2f} R, ${d['pnl']:+,.2f}")
            else:
                self.say(f"[B] {sym} limit missed: {d['result']}")
            row = {k: d.get(k) for k in TRADE_FIELDS}
            for k in ("signal_time", "fill_time", "exit_time"):
                row[k] = _iso(row[k])
            self._append("signal_trades.csv", TRADE_FIELDS, row)

    def _restore(self) -> None:
        path = self.reports / "signal_trades.csv"
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        for _, r in df.iterrows():
            eng = self.engines.get(r["symbol"])
            if eng is None:
                continue
            book = eng.books[r["strategy"]]
            setup = r["setup"] if "setup" in r and isinstance(r["setup"], str) else "unknown"
            if r["status"] == "missed":
                book.missed.append({"setup": setup})
            else:
                book.closed.append({"r": r["r"], "pnl": r["pnl"], "fees": r["fees"], "result": r["result"],
                                    "setup": setup})
                book.equity += float(r["pnl"])

    def view(self, symbol: str) -> dict:
        v = self.engines[symbol].view()
        v["rules"] = {"wall_min": self.rules.wall_min_for(self.engines[symbol].asset),
                      "imb_min": self.rules.imb_min, "flow_share": self.rules.flow_share,
                      "rr_min": self.rules.rr_min, "near_pct": self.rules.near_pct}
        v["recorded_files"] = self.recorder.files_written if self.recorder else 0
        return v
