"""Score the S/R Order-Flow Analyzer on recorded level visits.

Every visit (price within 0.1% of one of your levels) is saved with its
per-second order-book measurements and its outcome, which is decided WITHOUT
candles: did price go 0.3% through the level first (broke), 0.3% back first
(held), or neither within 60 min (unresolved)?

This script replays those seconds through the verdict rules and asks:
  * when it said BREAKOUT pressure, how often did the level break?
  * when it said REVERSAL pressure, how often did it hold?
  * compared with the base rate (how often levels broke or held anyway).
A call only counts if it was made BEFORE the outcome was decided.

Honest test: visits are split by time, the first 70% for development and
the last 30% for testing. With --grid, a few thresholds are tried on
development only, and the best is then scored once on test.

Versions compared on the same visits (one filter switched off at a time,
to see which filters really help): baseline rules, v2 with all filters,
and v2 without price response / persistence / failed-cross / quality gate.
Overlapping visits (another visit on the same coin was already running)
are left out, so one price move is not counted twice. Unresolved visits stay
in: a call followed by "unresolved" counts as not right.

Run:  python -m scripts.score_sr_analyzer          (default rules)
      python -m scripts.score_sr_analyzer --grid
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from app.config import PROJECT_ROOT
import json

from app.sranalyzer.measure import AnalyzerRules, classify, verdict, window
from app.sranalyzer.scoring import FailedCross, ScoreRules, features, scores, verdict2

DEV_SHARE = 0.7
MIN_CALLS = 10
GRID = {"consume_high": [0.15, 0.25, 0.40], "refill_high": [0.6, 0.8, 1.0], "attack_share": [0.55, 0.60, 0.65]}


def first_call(raw: pd.DataFrame, kind: str, level: float, rules: AnalyzerRules, until: float | None):
    recs = raw.to_dict("records")
    for i in range(len(recs)):
        t = recs[i]["t"]
        if until is not None and t >= until:
            break
        hist = recs[max(0, i - 70):i + 1]
        c15 = classify(window(hist, t, 15, rules), kind, level, rules)[0]
        c60 = classify(window(hist, t, 60, rules), kind, level, rules)[0]
        label, strength = verdict(c15, c60)
        if label != "inconclusive":
            return label, strength
    return "inconclusive", ""


def first_call_v2(raw: pd.DataFrame, kind: str, level: float, symbol: str, sr: ScoreRules,
                  until: float | None, rules: AnalyzerRules | None = None):
    """First STRONG verdict of the scored version (v2) before the outcome was decided."""
    rules = rules or AnalyzerRules()
    asset = symbol.replace("USDT.P", "").replace("USDT", "")
    recs = raw.to_dict("records")
    for r in recs:
        z = r.get("zone")
        r["zone"] = {float(k): v for k, v in json.loads(z).items()} if isinstance(z, str) else (z or {})
        r.setdefault("last", r["mid"])
    fc = FailedCross()
    for i in range(len(recs)):
        t = recs[i]["t"]
        if until is not None and t >= until:
            break
        hist = recs[max(0, i - 70):i + 1]
        w15, w60 = window(hist, t, 15, rules), window(hist, t, 60, rules)
        recs[i]["share15"] = w15["attack_share"] if w15 else None
        fc.update(t, recs[i]["last"], kind, level, recs[i]["share15"], sr)
        f = features(hist, w15, w60, kind, level, asset, fc.detected_at, t, sr)
        b, rv = scores(f, sr)
        label, _ = verdict2(f, b, rv, sr)
        if label in ("breakout", "reversal"):
            return label, "STRONG"
    return "inconclusive", ""


VARIANTS = {
    "baseline rules": None,
    "v2 (all filters)": ScoreRules(),
    "v2 without price response": ScoreRules(use_price_response=False),
    "v2 without liquidity/flow persistence": ScoreRules(use_persistence=False),
    "v2 without failed-cross": ScoreRules(use_failed_cross=False),
    "v2 without feed-quality gate": ScoreRules(use_quality_gate=False),
}


def score(visits: list[dict], rules: AnalyzerRules, sr: ScoreRules | None = None) -> dict:
    calls = []
    for v in visits:
        if sr is None:
            label, strength = first_call(v["raw"], v["kind"], v["level"], rules, v["resolved_t"])
        else:
            label, strength = first_call_v2(v["raw"], v["kind"], v["level"], v["symbol"], sr, v["resolved_t"],
                                            rules)
        calls.append({**v, "call": label, "strength": strength})
    n = len(calls)

    def rate(xs, outcome):
        return (sum(1 for x in xs if x["outcome"] == outcome) / len(xs)) if xs else None
    bo = [c for c in calls if c["call"] == "breakout"]
    rv = [c for c in calls if c["call"] == "reversal"]
    return {"visits": n, "base_broke": rate(calls, "broke"), "base_held": rate(calls, "held"),
            "unresolved": sum(1 for c in calls if c["outcome"] == "unresolved"),
            "breakout_calls": len(bo), "breakout_right": rate(bo, "broke"),
            "reversal_calls": len(rv), "reversal_right": rate(rv, "held"),
            "inconclusive": n - len(bo) - len(rv)}


def edge(s: dict) -> float | None:
    """Average improvement of the calls over the base rates (None if too few calls)."""
    parts = []
    if s["breakout_calls"] >= MIN_CALLS and s["base_broke"] is not None:
        parts.append(s["breakout_right"] - s["base_broke"])
    if s["reversal_calls"] >= MIN_CALLS and s["base_held"] is not None:
        parts.append(s["reversal_right"] - s["base_held"])
    return sum(parts) / len(parts) if parts else None


def fmt(s: dict) -> list[str]:
    p = lambda x: "n/a" if x is None else f"{x:.0%}"
    return [f"  {s['visits']} visits. Levels broke {p(s['base_broke'])}, held {p(s['base_held'])}, "
            f"unresolved {s['unresolved']} (base rates; unresolved visits are kept, they count as wrong calls)",
            f"  BREAKOUT calls: {s['breakout_calls']}, then broke {p(s['breakout_right'])}",
            f"  REVERSAL calls: {s['reversal_calls']}, then held {p(s['reversal_right'])}",
            f"  inconclusive: {s['inconclusive']}"]


def load(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for r in df.to_dict("records"):
        f = r.get("raw_file")
        if not isinstance(f, str) or not f or not Path(f).exists():
            continue
        rt = pd.Timestamp(r["resolved_time"]).timestamp() if isinstance(r.get("resolved_time"), str) and r["resolved_time"] else None
        if str(r.get("overlap", "")).lower() in ("true", "1"):
            continue                        # overlapping visit: same price move as another one
        out.append({"time": pd.Timestamp(r["time"]).timestamp(), "symbol": r["symbol"], "level": float(r["level"]),
                    "kind": r["kind"], "outcome": r["outcome"], "resolved_t": rt,
                    "raw": pd.read_parquet(f).sort_values("t")})
    return sorted(out, key=lambda v: v["time"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    path = PROJECT_ROOT / "reports" / "sr_visits.csv"
    lines = ["# S/R Order-Flow Analyzer score", ""]

    def say(s=""):
        print(s)
        lines.append(s)
    if not path.exists():
        say("No visits recorded yet (reports/sr_visits.csv). Keep the server running with your levels added.")
        return
    visits = load(path)
    if not visits:
        say("Visits are listed but their per-second files are missing (MM_ORDERBOOK_RECORD must be true).")
        return
    cut = visits[int(len(visits) * DEV_SHARE)]["time"] if len(visits) > 1 else visits[-1]["time"] + 1
    dev = [v for v in visits if v["time"] < cut]
    test = [v for v in visits if v["time"] >= cut]
    say(f"{len(visits)} visits with data: {len(dev)} development, {len(test)} test")
    if len(visits) < 50:
        say("Warning: fewer than 50 visits. Percentages can swing a lot by chance.")
    rules = AnalyzerRules()
    if args.grid and dev:
        best = None
        for vals in itertools.product(*GRID.values()):
            cand = replace(rules, **dict(zip(GRID, vals)))
            e = edge(score(dev, cand))
            say(f"  dev {dict(zip(GRID, vals))}: edge {'n/a' if e is None else f'{e:+.0%}'}")
            if e is not None and (best is None or e > best[0]):
                best = (e, cand)
        if best is None:
            say(f"  No setting made {MIN_CALLS}+ calls on development: keeping the defaults.")
        else:
            rules = best[1]
            say(f"  Chosen on development: consume_high {rules.consume_high}, refill_high {rules.refill_high}, "
                f"attack_share {rules.attack_share}")
    for name, part in (("Development (first 70%)", dev), ("Test (last 30%, untouched)", test)):
        say(f"## {name}")
        if not part:
            say("  no visits")
            continue
        for sym in sorted({v["symbol"] for v in part}):
            say(f"### {sym}")
            sv = [v for v in part if v["symbol"] == sym]
            for vname, sr in VARIANTS.items():
                say(f"  [{vname}]")
                for line in fmt(score(sv, rules, sr)):
                    say("  " + line)
    say()
    say("A call is only useful if its hit rate is clearly ABOVE the base rate on the TEST visits. "
        "With few calls, a difference of 10-20 points can still be luck.")
    out = PROJECT_ROOT / "reports" / "sr_analyzer_score.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
