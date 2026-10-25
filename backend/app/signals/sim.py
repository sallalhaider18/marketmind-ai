"""Paper positions and result statistics for the two entry strategies.

A = confirmed reaction, marketable entry (taker fee + slippage).
B = confirmed setup, then a limit order at the retest. Filled only if price
    trades THROUGH the limit (touching it is not enough: we assume we are at
    the back of the queue). If price runs to the target first, or 60 min pass,
    the order is "missed".
Exits: stop (taker + slippage, checked first), target (limit, maker), or the
24 h time limit (taker).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from app.signals.rules import Rules


@dataclass
class Position:
    strategy: str             # "A" or "B"
    symbol: str
    side: str                 # "long" / "short"
    level: float
    entry: float              # A: fill price; B: limit price
    stop: float
    target: float
    signal_time: float
    status: str = "pending"   # pending -> open -> closed / missed
    qty: float = 0.0          # coins
    fill_time: float | None = None
    exit_time: float | None = None
    exit_price: float | None = None
    result: str | None = None      # target / stop / time / missed reason
    fees: float = 0.0              # $ paid
    pnl: float = 0.0               # $ after costs
    r: float = 0.0                 # pnl / planned risk
    risk_usd: float = 0.0
    setup: str = ""                # support bounce / support breakdown / resistance rejection / resistance breakout

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1

    def as_dict(self) -> dict:
        return asdict(self)


def plan_rr(side: str, entry: float, stop: float, target: float, fee_in: float, rules: Rules) -> float | None:
    """Reward : risk per coin after fees and slippage (target = maker exit, stop = taker + slip)."""
    s = 1 if side == "long" else -1
    reward = s * (target - entry) - entry * fee_in - target * rules.maker
    stop_fill = stop * (1 - s * rules.slip)
    risk = s * (entry - stop_fill) + entry * fee_in + stop_fill * rules.taker
    if risk <= 0 or s * (target - entry) <= 0:
        return None
    return reward / risk


def open_position(p: Position, equity: float, rules: Rules, t: float, fee_in: float) -> None:
    """Size for 1% risk of this strategy's account and mark it open."""
    per_coin = abs(p.entry - p.stop)
    p.risk_usd = equity * rules.risk_pct
    p.qty = p.risk_usd / per_coin if per_coin > 0 else 0.0
    p.status, p.fill_time = "open", t
    p.fees = p.qty * p.entry * fee_in


def close_position(p: Position, price: float, fee: float, t: float, why: str) -> None:
    p.status, p.exit_time, p.exit_price, p.result = "closed", t, price, why
    p.fees += p.qty * price * fee
    p.pnl = p.sign * (price - p.entry) * p.qty - p.fees
    p.r = p.pnl / p.risk_usd if p.risk_usd else 0.0


def update_position(p: Position, row: dict, rules: Rules, equity: float) -> str | None:
    """Advance one second of market data. Returns an event name or None."""
    t, hi, lo, mid = row["t"], row["hi"], row["lo"], row["mid"]
    s = p.sign
    if p.status == "pending":                       # strategy B limit order
        through = lo < p.entry if s > 0 else hi > p.entry
        if through:
            open_position(p, equity, rules, t, rules.maker)
            # same second also through the stop? (conservative: yes, stopped)
            if (lo <= p.stop) if s > 0 else (hi >= p.stop):
                close_position(p, p.stop * (1 - s * rules.slip), rules.taker, t, "stop")
                return "fill+close"
            return "fill"
        if (hi > p.target) if s > 0 else (lo < p.target):
            p.status, p.result, p.exit_time = "missed", "price reached the target before the limit filled", t
            return "missed"
        if t - p.signal_time >= rules.limit_expiry:
            p.status, p.result, p.exit_time = "missed", "not filled within 60 min", t
            return "missed"
        return None
    if p.status != "open":
        return None
    if (lo <= p.stop) if s > 0 else (hi >= p.stop):
        # a gap past the stop fills at the worse of stop and current price
        px = min(p.stop, mid) if s > 0 else max(p.stop, mid)
        close_position(p, px * (1 - s * rules.slip), rules.taker, t, "stop")
        return "close"
    if (hi > p.target) if s > 0 else (lo < p.target):
        close_position(p, p.target, rules.maker, t, "target")
        return "close"
    if t - (p.fill_time or t) >= rules.max_hold:
        close_position(p, mid * (1 - s * rules.slip), rules.taker, t, "time")
        return "close"
    return None


def summarize(trades: list[dict], missed: int = 0) -> dict:
    """Net results of closed trades (dicts with r, pnl, fees, result)."""
    n = len(trades)
    out = {"trades": n, "missed": missed,
           "fill_rate": (n / (n + missed)) if (n + missed) else None}
    if not n:
        return out
    rs = [float(x["r"]) for x in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    mean = sum(rs) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1)) if n > 1 else 0.0
    curve, peak, dd = 0.0, 0.0, 0.0
    for r in rs:
        curve += r
        peak = max(peak, curve)
        dd = max(dd, peak - curve)
    gross_win, gross_loss = sum(wins), -sum(losses)
    out.update({
        "win_rate": len(wins) / n,
        "avg_win_r": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss_r": (-gross_loss / len(losses)) if losses else 0.0,
        "expectancy_r": mean,
        "ci95_r": 1.96 * sd / math.sqrt(n) if n > 1 else None,
        "total_r": sum(rs),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_drawdown_r": dd,
        "net_pnl": sum(float(x["pnl"]) for x in trades),
        "fees": sum(float(x["fees"]) for x in trades),
    })
    return out
