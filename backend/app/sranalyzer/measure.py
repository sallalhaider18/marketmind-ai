"""S/R order-flow measurements (no candles, no indicators).

For one level, every second:
  * displayed depth on the DEFENDING side near the level
    (asks at a resistance, bids at a support) within 0.1% and 0.25%;
  * executed: aggressive volume that traded INTO that defending depth;
  * how the displayed depth changed, split into
      consumed     = removed by trades   (min(removed, executed))
      cancelled    = removed without trades (removed - executed, if positive)
      replenished  = new size added, plus "hidden" refills
                     (executed more than was removed = an iceberg refilling);
    over a window, cancels and new orders are NETTED against each other:
    market makers cancel and re-post all the time, which is neither a pull
    nor a refill;
  * attack / defence: all aggressive volume pushing toward the level
    (buys at a resistance, sells at a support) and against it.

Windows of 5, 15 and 60 seconds add these up. The verdict:
  BREAKOUT pressure   attackers in control, defending depth is being CONSUMED,
                      little replenishment, depth falling, price not retreating
  REVERSAL pressure   attackers in control and trading into the depth, but it is
                      REPLENISHED, depth is not falling and price is stalling
  INCONCLUSIVE        anything else (no clear attacker, depth pulled rather than
                      traded, nothing displayed, conflicting signs)
HIGH = the 15 s and 60 s windows agree; MODERATE = only the 60 s window.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalyzerRules:
    near_pct: float = 0.001          # "at the level" depth range: 0.1%
    wide_pct: float = 0.0025         # wider range: 0.25% (used for the flow measures)
    watch_pct: float = 0.01          # measure a level while price is within 1% of it
    visit_pct: float = 0.001         # a VISIT starts when price comes within 0.1%
    min_coverage: float = 0.8        # a window needs data for 80% of its seconds
    attack_share: float = 0.60       # attackers' share of aggressive volume
    consume_high: float = 0.25       # >= 25% of the displayed defence traded away per minute
    consume_low: float = 0.10
    refill_high: float = 0.80        # replenished >= 80% of what was consumed
    refill_low: float = 0.50
    depth_drop: float = 0.25         # breakout: defending depth down >= 25%
    depth_hold: float = 0.10         # reversal: defending depth down <= 10%
    stall_pct: float = 0.02          # reversal: price moved <= 0.02% toward/through the level
    outcome_pct: float = 0.003       # visit outcome: 0.3% through (broke) or back (held) first ...
    horizon: float = 3600            # ... within 60 min


def defend_side(kind: str) -> str:
    return "asks" if kind == "resistance" else "bids"


def zone(levels: dict, level: float, pct: float) -> dict:
    lo, hi = level * (1 - pct), level * (1 + pct)
    return {p: q for p, q in levels.items() if lo <= p <= hi}


def second_record(t: float, mid: float, kind: str, level: float, prev_zone: dict | None, zone_now: dict,
                  trades: list, cs: float, rules: AnalyzerRules) -> dict:
    """One second of measurements. zone_now / prev_zone: defending-side {price: contracts} within wide_pct."""
    d = 1 if kind == "resistance" else -1
    attack_side = "buy" if d > 0 else "sell"
    lo, hi = level * (1 - rules.wide_pct), level * (1 + rules.wide_pct)
    executed = sum(x.contracts for x in trades if x.side == attack_side and lo <= x.price <= hi) * cs
    attack = sum(x.contracts for x in trades if x.side == attack_side) * cs
    defence = sum(x.contracts for x in trades if x.side != attack_side) * cs
    removed = added = 0.0
    if prev_zone is not None:
        for p in set(prev_zone) | set(zone_now):
            diff = zone_now.get(p, 0.0) - prev_zone.get(p, 0.0)
            if diff < 0:
                removed -= diff
            else:
                added += diff
    removed *= cs
    added *= cs
    consumed = min(removed, executed)
    nlo, nhi = level * (1 - rules.near_pct), level * (1 + rules.near_pct)
    return {
        "t": t, "mid": mid,
        "depth10": sum(q for p, q in zone_now.items() if nlo <= p <= nhi) * cs,
        "depth25": sum(zone_now.values()) * cs,
        "executed": executed,
        "consumed": consumed,
        "cancelled": max(0.0, removed - executed),
        "replenished": added + max(0.0, executed - removed),
        "attack": attack, "defence": defence,
        "known": prev_zone is not None,
    }


def window(records: list[dict], now: float, seconds: int, rules: AnalyzerRules) -> dict | None:
    rs = [r for r in records if r["t"] > now - seconds]
    if len(rs) < rules.min_coverage * seconds or not rs:
        return None
    known = [r for r in rs if r["known"]]
    tot = lambda k: sum(r[k] for r in known)
    attack, defence = sum(r["attack"] for r in rs), sum(r["defence"] for r in rs)
    first, last = rs[0], rs[-1]
    return {
        "seconds": seconds,
        "depth_start": first["depth25"], "depth_end": last["depth25"], "depth10": last["depth10"],
        "executed": tot("executed"), "consumed": tot("consumed"), "cancelled": tot("cancelled"),
        "replenished": tot("replenished"),
        "attack": attack, "defence": defence,
        "attack_share": attack / (attack + defence) if attack + defence > 0 else None,
        "cvd": attack - defence,                      # positive = net aggressive volume INTO the level
        "mid_start": first["mid"], "mid_end": last["mid"],
    }


def net_refill(w: dict) -> float:
    """New size added beyond what was cancelled in the same window (market makers cancel and
    re-post constantly; that churn is neither a pull nor a refill), plus hidden refills."""
    return max(0.0, w["replenished"] - w["cancelled"])


def classify(w: dict | None, kind: str, level: float, rules: AnalyzerRules) -> tuple[str, str]:
    """('breakout' | 'reversal' | 'inconclusive', reason)."""
    if w is None:
        return "inconclusive", "not enough data yet"
    d = 1 if kind == "resistance" else -1
    share = w["attack_share"]
    attackers = "buyers" if d > 0 else "sellers"
    if share is None or share < rules.attack_share:
        return "inconclusive", f"{attackers} are not clearly attacking the level"
    if w["cvd"] <= 0:
        return "inconclusive", "CVD is not moving into the level"
    if w["depth_start"] <= 0 and w["executed"] <= 0:
        return "inconclusive", "nothing displayed at the level to measure"
    # amounts are scaled to "per minute" so 15 s and 60 s windows use the same thresholds
    per_min = 60 / w["seconds"]
    base = max(w["depth_start"], 1e-12) / per_min
    consumption = w["consumed"] / base
    refilled = net_refill(w)
    refill = refilled / w["consumed"] if w["consumed"] > 0 else (1.0 if refilled > 0 else 0.0)
    net = (w["depth_end"] - w["depth_start"]) / base
    progress = d * (w["mid_end"] - w["mid_start"]) / level * 100
    if net <= -rules.depth_drop and consumption < rules.consume_low:
        return "inconclusive", "the defence was pulled (cancelled) rather than traded"
    if (consumption >= rules.consume_high and refill < rules.refill_low and net <= -rules.depth_drop
            and progress >= 0):
        return "breakout", "defending orders are being consumed and not refilled"
    if (w["executed"] / base >= rules.consume_high and refill >= rules.refill_high
            and net >= -rules.depth_hold and progress <= rules.stall_pct):
        return "reversal", "attackers trade into the level but it keeps refilling and price stalls"
    return "inconclusive", "mixed evidence"


def readings(w: dict | None, kind: str, level: float, rules: AnalyzerRules) -> dict | None:
    """Plain-word readings for the dashboard (from one window)."""
    if w is None:
        return None
    d = 1 if kind == "resistance" else -1
    base = max(w["depth_start"], 1e-12)
    consumption = w["consumed"] / base if w["depth_start"] > 0 else None
    refill = net_refill(w) / w["consumed"] if w["consumed"] > 0 else None
    share = w["attack_share"]
    net_pull = max(0.0, w["cancelled"] - w["replenished"])

    def grade(v, hi, lo, words):
        if v is None:
            return "—"
        return words[0] if v >= hi else words[2] if v < lo else words[1]
    return {
        "consumption": grade(consumption, rules.consume_high, rules.consume_low, ("High", "Neutral", "Low")),
        "consumption_pct": consumption,
        "attack": grade(share, rules.attack_share, 0.5, ("Strong", "Moderate", "Weak")),
        "attack_share": share,
        "replenishment": grade(refill, rules.refill_high, rules.refill_low, ("High", "Medium", "Low")),
        "refill_ratio": refill,
        "cvd": "Rising into the level" if w["cvd"] > 0 else "Falling (away from the level)",
        "cvd_value": w["cvd"],
        "cancelled_pct": net_pull / base if w["depth_start"] > 0 else None,      # net of re-posting
        "depth_change_pct": (w["depth_end"] - w["depth_start"]) / base if w["depth_start"] > 0 else None,
        "progress_pct": d * (w["mid_end"] - w["mid_start"]) / level * 100,
    }


def verdict(c15: str, c60: str) -> tuple[str, str]:
    """(label, strength) from the 15 s and 60 s classifications."""
    if c60 in ("breakout", "reversal"):
        if c15 == c60:
            return c60, "HIGH"
        if c15 == "inconclusive":
            return c60, "MODERATE"
    return "inconclusive", ""
