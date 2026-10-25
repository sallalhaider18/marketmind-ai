"""Replay the recorded order-flow data through the signal engine (strategies A and B).

Uses the one-second rows the server saves while it runs
(data/signals/mexc_futures/<SYMBOL>/...). The engine is the same code that
runs live, so this tests exactly what would have been traded.

Honest test:
  * the first 70% of the recorded time is "development", the last 30% is
    "test" and is never used to choose anything;
  * with --grid, a few thresholds are tried on development only; the best one
    (by strategy A expectancy, at least 20 trades) is then run once on test;
  * results are after fees (0% maker, 0.02% taker) and 0.01% slippage;
    strategy B fills only when price trades through its limit.
Levels at each moment use only 4h / daily candles that had already closed.

Run:  python -m scripts.replay_signals            (default rules)
      python -m scripts.replay_signals --grid     (also try a few thresholds)
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from app.config import PROJECT_ROOT, get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.obtrade.levels import find_levels
from app.signals.engine import SETUP_NAMES, SignalEngine
from app.signals.runner import level_source as source_name
from app.signals.rows import load_rows
from app.signals.rules import Rules

DEV_SHARE = 0.7
MIN_DEV_TRADES = 20
GRID = {"imb_min": [0.15, 0.25, 0.35], "flow_share": [0.55, 0.60, 0.65]}


def level_source(df4: pd.DataFrame, df1: pd.DataFrame):
    """levels_fn(t, mid): strong levels from candles fully closed before t, recomputed every 30 min."""
    cache: dict[int, list[float]] = {}
    sources: dict[float, str] = {}
    t4 = df4["timestamp"].astype("int64") // 10**9 if len(df4) else pd.Series(dtype="int64")
    t1 = df1["timestamp"].astype("int64") // 10**9 if len(df1) else pd.Series(dtype="int64")

    def fn(t: float, mid: float | None) -> list[float]:
        k = int(t // 1800)
        if k not in cache:
            a = df4[(t4 + 4 * 3600) <= t] if len(df4) else df4
            b = df1[(t1 + 86400) <= t] if len(df1) else df1
            found = find_levels(a, b, current=mid)
            cache[k] = [lv.price for lv in found]
            sources.update({lv.price: source_name(lv) for lv in found})
        return cache[k]
    fn.sources = sources
    return fn


def run(symbol: str, rows: list[dict], levels, rules: Rules) -> dict:
    asset = symbol.replace("USDT.P", "")
    holder = {"mid": None}
    evals: list[dict] = []
    tests: list[dict] = []

    def on(k, d):
        if k == "evaluation":
            evals.append(d)
        elif k == "level_test":
            tests.append(d)
    srcs = getattr(levels, "sources", {})
    eng = SignalEngine(symbol, asset, lambda t: levels(t, holder["mid"]), rules, on_event=on,
                       level_info=lambda p: srcs.get(p, ""))
    for r in rows:
        if r.get("mid"):
            holder["mid"] = r["mid"]
        eng.on_row(r)
    st = eng.stats()
    checks = [e for e in evals if "ok_wall" in e]
    st["evaluations"] = len(checks)
    st["signals"] = sum(1 for e in evals if e["passed"])
    st["pass_rates"] = {k: (sum(1 for e in checks if e.get(f"ok_{k}")) / len(checks)) if checks else None
                        for k in ("wall", "imbalance", "flow", "cvd", "price", "risk")}
    st["level_tests"] = tests
    return st


def level_test_report(tests: list[dict]) -> list[str]:
    """How often levels broke vs held, overall and split by what CVD / the wall did at the touch."""
    if not tests:
        return ["  level tests: none finished"]
    n = len(tests)

    def rates(xs):
        k = len(xs)
        if not k:
            return "n/a"
        b = sum(x["outcome"] == "broke" for x in xs)
        h = sum(x["outcome"] == "held" for x in xs)
        return f"{k} tests: broke {b / k:.0%}, held {h / k:.0%}, unresolved {(k - b - h) / k:.0%}"

    def toward(x):          # 5-min CVD at the touch pushing INTO the level (sellers at support, buyers at resistance)
        c = x.get("at_touch_cvd_5m")
        if c is None or c != c:
            return None
        return (c < 0) if x["kind"] == "support" else (c > 0)
    out = [f"  level tests: {rates(tests)}"]
    for kind in ("support", "resistance"):
        out.append(f"    {kind}: {rates([x for x in tests if x['kind'] == kind])}")
    out.append(f"    5-min CVD pushing into the level at the touch: {rates([x for x in tests if toward(x) is True])}")
    out.append(f"    5-min CVD pushing away at the touch:           {rates([x for x in tests if toward(x) is False])}")
    for fate in ("held", "absorbing", "consumed", "pulled", "none"):
        xs = [x for x in tests if x.get("wall_fate") == fate]
        if xs:
            out.append(f"    wall {fate}: {rates(xs)}")
    if n < 50:
        out.append("    (fewer than 50 tests: these percentages can swing a lot by chance)")
    return out


def setup_report(st: dict) -> list[str]:
    out = []
    for name in SETUP_NAMES.values():
        row = st["by_setup"][name]
        bits = []
        for k in ("A", "B"):
            s = row[k]
            bits.append(f"{k} {s.get('trades', 0)} trades" + (f", {s['expectancy_r']:+.2f} R avg" if s.get("trades") else ""))
        out.append(f"    {name}: " + "; ".join(bits))
    return out


def fmt(st: dict, k: str) -> str:
    s = st[k]
    if not s.get("trades"):
        extra = f", {s['missed']} missed" if k == "B" and s.get("missed") else ""
        return f"  {k}: no trades{extra}"
    pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
    ci = "" if s["ci95_r"] is None else f" ± {s['ci95_r']:.2f}"
    fill = f", fill rate {s['fill_rate']:.0%}" if k == "B" else ""
    return (f"  {k}: {s['trades']} trades, win {s['win_rate']:.0%}, expectancy {s['expectancy_r']:+.2f} R{ci}, "
            f"total {s['total_r']:+.1f} R, PF {pf}, max DD {s['max_drawdown_r']:.1f} R, "
            f"avg win {s['avg_win_r']:+.2f} / loss {s['avg_loss_r']:+.2f} R, fees ${s['fees']:,.0f}{fill}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    base = settings.data_dir / "signals" / "mexc_futures"
    symbols = args.symbols or settings.symbols
    lines = ["# Order-flow signal replay", ""]

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    for sym in symbols:
        rows = load_rows(base, sym)
        if not rows:
            say(f"{sym}: no recorded data in {base / sym}. Run the server for a while first "
                "(MM_ORDERBOOK_RECORD=true).")
            continue
        span_h = (rows[-1]["t"] - rows[0]["t"]) / 3600
        valid = sum(1 for r in rows if r.get("valid"))
        say(f"## {sym}: {len(rows):,} seconds recorded over {span_h:.1f} h ({valid / len(rows):.0%} valid)")
        if span_h < 14 * 24:
            say("  Warning: less than 2 weeks of data. Treat any result as anecdote, not evidence.")
        df4 = load_candles(candle_path(settings.raw_dir, "mexc_futures", sym, "4h"))
        df1 = load_candles(candle_path(settings.raw_dir, "mexc_futures", sym, "1d"))
        if df4.empty or df1.empty:
            from app.adapters.mexc_futures import MexcFuturesAdapter
            ad = MexcFuturesAdapter(settings.mexc_rest_url)
            df4 = update_history(ad, sym, "4h", settings.raw_dir, days=730)[1]
            df1 = update_history(ad, sym, "1d", settings.raw_dir, days=1095)[1]
        levels = level_source(df4, df1)
        cut = rows[0]["t"] + (rows[-1]["t"] - rows[0]["t"]) * DEV_SHARE
        dev = [r for r in rows if r["t"] < cut]
        test = [r for r in rows if r["t"] >= cut]
        rules = Rules()
        if args.grid:
            best = None
            for vals in itertools.product(*GRID.values()):
                cand = replace(rules, **dict(zip(GRID, vals)))
                st = run(sym, dev, levels, cand)
                a = st["A"]
                say(f"  dev grid {dict(zip(GRID, vals))}: A {a.get('trades', 0)} trades, "
                    f"expectancy {a.get('expectancy_r', float('nan')):+.2f} R")
                if a.get("trades", 0) >= MIN_DEV_TRADES and (best is None or a["expectancy_r"] > best[0]):
                    best = (a["expectancy_r"], cand)
            if best is None:
                say(f"  No setting reached {MIN_DEV_TRADES} development trades: keeping the default rules.")
            else:
                rules = best[1]
                say(f"  Chosen on development: imb_min {rules.imb_min}, flow_share {rules.flow_share}")
        for name, part in (("Development (first 70%)", dev), ("Test (last 30%, untouched)", test)):
            st = run(sym, part, levels, rules)
            pr = ", ".join(f"{k} {v:.0%}" for k, v in st["pass_rates"].items() if v is not None)
            say(f"### {name}: {st['evaluations']} checks, {st['signals']} signals")
            if pr:
                say(f"  how often each check passed: {pr}")
            say(fmt(st, "A"))
            say(fmt(st, "B"))
            say("  by setup type:")
            for line in setup_report(st):
                say(line)
            for line in level_test_report(st["level_tests"]):
                say(line)
        say()
    say("Judge on the TEST period, net of costs: expectancy and its ± range, drawdown, and number of trades. "
        "A positive result with fewer than ~30 trades, or a ± range that includes 0, is not proof of an edge.")
    out = PROJECT_ROOT / "reports" / "signal_replay.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
