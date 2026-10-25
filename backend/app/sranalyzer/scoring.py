"""Scored S/R verdict (version 2), alongside the baseline rules in measure.py.

Extra evidence (all from the order book and executed trades, no candles):
  * PRICE RESPONSE: how far price moved toward/through the level per $1M of net
    aggressive volume, compared with this coin's normal response over the last
    hour. Lots of aggression with little movement = absorbed.
  * LIQUIDITY PERSISTENCE: of the size resting at the defending prices at the
    start of the window, how much is still at those SAME prices at the end
    (refills at the same price count; size moved elsewhere does not).
  * FAILED CROSS: price traded through the level, came back to the original
    side within 5 min, and then the defenders took over the flow. A failed
    breakout/breakdown is reversal evidence.
  * FEED QUALITY: book resynced in the last minute, trades arriving late, the
    trade stream gone quiet, or too little activity -> INSUFFICIENT DATA.
  * FLOW PERSISTENCE: in how many of the last 15 seconds the attackers held
    >= 60% of the 15-second flow (instead of "15 s and 60 s windows agree",
    which partly repeats the same trades).

Two independent scores 0-100 (they need not add up to 100):
  breakout = 30 flow + 30 liquidity giving way + 25 price moving + 15 flow persistence
  reversal = 30 flow + 30 liquidity holding    + 25 price failing + 15 flow persistence
These weights are a starting guess to be tested, not tuned values.

Verdicts: INSUFFICIENT DATA / STRONG BREAKOUT / STRONG REVERSAL /
BREAKOUT LEAN / REVERSAL LEAN / CONFLICTING / INCONCLUSIVE.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from app.sranalyzer.measure import net_refill


@dataclass
class ScoreRules:
    strong: float = 75               # score for a STRONG verdict
    at_level_pct: float = 0.001      # STRONG needs price within 0.1% of the level (it is being tested)
    hit_min: float = 0.25            # STRONG reversal: attackers traded >= 25%/min of the displayed defence
    lean: float = 60                 # score for a LEAN verdict; both >= this = CONFLICTING
    persist_min: float = 0.6         # strong breakout needs flow held in >= 60% of the last 15 s
    # feed quality
    max_trade_lag: float = 3.0       # trades arriving more than 3 s after they happened
    max_trade_gap: float = 30.0      # no trade at all for 30 s
    resync_quiet: float = 60.0       # book rebuilt within the last 60 s
    min_quality: float = 0.9         # >= 90% of the window's seconds must be clean
    min_activity: dict = field(default_factory=lambda: {"BTC": 1.0, "ETH": 10.0})   # coins traded per 60 s
    # failed cross
    cross_pct: float = 0.0005        # traded 0.05% through the level ...
    return_pct: float = 0.0002       # ... then back 0.02% on the original side ...
    cross_window: float = 300        # ... within 5 min ...
    flip_window: float = 60          # ... and within 60 s the defenders hold >= 60% of 15-s flow
    failed_memory: float = 180       # a failed cross counts as evidence for 3 min
    # which evidence groups are used (switch off to test one filter at a time)
    use_price_response: bool = True
    use_persistence: bool = True
    use_failed_cross: bool = True
    use_quality_gate: bool = True


def clamp(x: float | None) -> float:
    if x is None:
        return 0.0
    return max(0.0, min(1.0, x))


class ImpactBaseline:
    """Normal price response of one coin: bp of price move per $1M of net aggressive volume,
    per completed minute, median over the last hour."""

    def __init__(self):
        self.minute: int | None = None
        self.start_mid: float | None = None
        self.net_usd = 0.0
        self.values: list[float] = []

    def add(self, t: float, mid: float, buy_coins: float, sell_coins: float) -> None:
        m = int(t // 60)
        if m != self.minute:
            if self.minute is not None and self.start_mid and abs(self.net_usd) >= 50_000:
                bp = abs(mid / self.start_mid - 1) * 1e4
                self.values.append(bp / (abs(self.net_usd) / 1e6))
                self.values = self.values[-60:]
            self.minute, self.start_mid, self.net_usd = m, mid, 0.0
        self.net_usd += (buy_coins - sell_coins) * mid

    def value(self) -> float | None:
        return statistics.median(self.values) if len(self.values) >= 10 else None


def survival(z_start: dict, z_end: dict) -> float | None:
    """Share of the starting resting size still at the same prices."""
    total = sum(z_start.values())
    if total <= 0:
        return None
    return sum(min(q, z_end.get(p, 0.0)) for p, q in z_start.items()) / total


def features(records: list[dict], w15: dict | None, w60: dict | None, kind: str, level: float,
             asset: str, failed_at: float | None, now: float, sr: ScoreRules) -> dict:
    """Normalised evidence (0..1) plus the raw numbers behind it."""
    d = 1 if kind == "resistance" else -1
    f: dict = {"quality_ok": True, "quality_reason": ""}
    rs60 = [r for r in records if r["t"] > now - 60]
    if sr.use_quality_gate:
        bad = [r for r in rs60 if not r.get("q_ok", True)]
        if w60 is None:
            f.update(quality_ok=False, quality_reason="less than a minute of clean data")
        elif rs60 and len(bad) / len(rs60) > 1 - sr.min_quality:
            f.update(quality_ok=False, quality_reason=bad[-1].get("q_reason") or "data problems")
        elif (w60["attack"] + w60["defence"]) < sr.min_activity.get(asset, 0.0):
            f.update(quality_ok=False, quality_reason="too little trading to judge")
    if w60 is None:
        return f
    share = w60["attack_share"]
    base = max(w60["depth_start"], 1e-12)
    consumption = w60["consumed"] / base if w60["depth_start"] > 0 else None
    refill = (net_refill(w60) / w60["consumed"]) if w60["consumed"] > 0 else None
    net = (w60["depth_end"] - w60["depth_start"]) / base if w60["depth_start"] > 0 else None
    progress = d * (w60["mid_end"] - w60["mid_start"]) / level * 100            # % toward/through
    surv = survival(rs60[0].get("zone", {}), rs60[-1].get("zone", {})) if rs60 else None
    # price response vs this coin's normal
    impact_ratio = None
    basev = rs60[-1].get("impact_base") if rs60 else None
    cvd_usd = w60["cvd"] * w60["mid_end"]
    if basev and cvd_usd > 50_000:
        eff = max(progress, 0.0) * 100 / (cvd_usd / 1e6)                           # bp per $1M
        impact_ratio = eff / basev
    # flow persistence: last 15 s where the 15-s attackers' share >= 60%
    held = [r.get("share15") for r in records if r["t"] > now - 15 and r.get("share15") is not None]
    persist = (sum(1 for s in held if s >= 0.6) / len(held)) if held else 0.0
    failed = failed_at is not None and now - failed_at <= sr.failed_memory
    hit = (w60["executed"] / base) * (60 / w60["seconds"]) if w60["depth_start"] > 0 else None
    f.update({
        "distance_pct": abs(w60["mid_end"] - level) / level * 100,
        "hit_ratio": hit,
        "attack_share": share, "consumption": consumption, "refill": refill, "depth_change": net,
        "progress_pct": progress, "survival": surv, "impact_ratio": impact_ratio, "persistence": persist,
        "failed_cross": failed,
        # normalised 0..1
        "n_flow": clamp(((share or 0.5) - 0.5) / 0.25) if w60["cvd"] > 0 else 0.0,
        "n_consume": clamp((consumption or 0.0) / 0.5),
        "n_no_refill": 1 - clamp(refill if refill is not None else 1.0),
        "n_depth_fall": clamp(-(net or 0.0) / 0.5),
        "n_refill": clamp(refill if refill is not None else 0.0),
        "n_survive": clamp(surv),
        "n_progress": clamp(progress / 0.05),
        "n_stall": 1 - clamp(progress / 0.05),
        "n_impact_high": clamp(((impact_ratio or 0.0) - 0.5) / 1.0) if impact_ratio is not None else None,
        "n_absorbed": clamp((1 - impact_ratio) / 0.8) if impact_ratio is not None else None,
    })
    return f


def _mean(xs: list) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def scores(f: dict, sr: ScoreRules) -> tuple[float | None, float | None]:
    if "n_flow" not in f:
        return None, None
    persist = f["persistence"] if sr.use_persistence else 0.0
    wp = 0.15 if sr.use_persistence else 0.0
    giving = [f["n_consume"], f["n_no_refill"], f["n_depth_fall"]]
    holding = [f["n_refill"], 1 - f["n_depth_fall"]]
    if sr.use_persistence:
        giving.append(1 - f["n_survive"])
        holding.append(f["n_survive"])
    moving = [f["n_progress"]] + ([f["n_impact_high"]] if sr.use_price_response else [])
    failing = [f["n_stall"]] + ([f["n_absorbed"]] if sr.use_price_response else [])
    flow_r = f["n_flow"]
    if sr.use_failed_cross and f["failed_cross"]:
        failing = [1.0]
        flow_r = max(flow_r, 0.8)              # the attack happened and failed
    total = 0.85 + wp
    b = (30 * f["n_flow"] + 30 * _mean(giving) + 25 * _mean(moving) + 100 * wp * persist) / total
    r = (30 * flow_r + 30 * _mean(holding) + 25 * _mean(failing) + 100 * wp * persist) / total
    return round(b, 1), round(r, 1)


def verdict2(f: dict, b: float | None, r: float | None, sr: ScoreRules) -> tuple[str, str]:
    """(label, reason). Labels: insufficient, breakout, reversal, breakout_lean, reversal_lean,
    conflicting, inconclusive."""
    if not f.get("quality_ok", True):
        return "insufficient", f["quality_reason"]
    if b is None:
        return "insufficient", "not enough data yet"
    if b >= sr.lean and r >= sr.lean:
        return "conflicting", f"breakout {b:.0f} and reversal {r:.0f} are both high"
    at_level = f["distance_pct"] <= sr.at_level_pct * 100 or f["failed_cross"]
    if not at_level and max(b, r) >= sr.lean:
        side = "breakout" if b >= r else "reversal"
        return f"{side}_lean", (f"{side} score {max(b, r):.0f}, but price is {f['distance_pct']:.2f}% away: "
                                "the level is not being tested yet")
    if (b >= sr.strong and f["progress_pct"] > 0
            and (f["persistence"] >= sr.persist_min or not sr.use_persistence)):
        return "breakout", f"breakout score {b:.0f}: attackers in control, defence giving way, price moving"
    if sr.use_price_response and f.get("n_absorbed") is not None:
        absorbed = f["n_absorbed"] >= 0.5
    else:                                   # no normal-response baseline yet (first 10 min): price must clearly stall
        absorbed = f["n_stall"] >= 0.8
    hitting = (f.get("hit_ratio") or 0) >= sr.hit_min
    if r >= sr.strong and ((absorbed and hitting) or f["failed_cross"]) and f["progress_pct"] <= 0.02:
        why = "failed cross: price went through, came back and the flow turned" if f["failed_cross"] \
            else "aggression is absorbed: the level holds and price is not moving"
        return "reversal", f"reversal score {r:.0f}: {why}"
    if b >= sr.lean:
        return "breakout_lean", f"breakout score {b:.0f}, not strong yet"
    if r >= sr.lean:
        return "reversal_lean", f"reversal score {r:.0f}, not strong yet"
    return "inconclusive", f"breakout {b:.0f} / reversal {r:.0f}: no side is convincing"


class FailedCross:
    """Per level: through -> back within 5 min -> defenders take the flow within 60 s."""

    def __init__(self):
        self.crossed_at: float | None = None
        self.returned_at: float | None = None
        self.detected_at: float | None = None

    def update(self, t: float, price: float, kind: str, level: float, share15: float | None,
               sr: ScoreRules) -> bool:
        d = 1 if kind == "resistance" else -1
        through = d * (price - level) / level
        if through >= sr.cross_pct:
            self.crossed_at, self.returned_at = t, None
        elif self.crossed_at is not None and t - self.crossed_at > sr.cross_window and self.returned_at is None:
            self.crossed_at = None
        if self.crossed_at is not None and self.returned_at is None and through <= -sr.return_pct:
            self.returned_at = t
        if self.returned_at is not None:
            if t - self.returned_at > sr.flip_window:
                self.crossed_at = self.returned_at = None
            elif share15 is not None and share15 <= 0.4:
                self.detected_at = t
                self.crossed_at = self.returned_at = None
                return True
        return False
