"""Bar-by-bar backtester for one setup on one instrument.

Conservative by design:
  * Signals are read at a bar's close; orders act from the NEXT bar.
  * Market entries fill at the next bar's open plus slippage (taker fee).
  * Limit entries fill only if price trades at or through the limit, at the
    limit or a better open (maker fee). They expire after N bars; unfilled
    orders are counted, never treated as trades.
  * If the stop and the target are both inside one bar, the STOP is assumed
    hit first. A limit entry cannot take profit on the bar it filled.
  * Gaps through the stop exit at the (worse) open.
  * Stops and time exits pay taker fee + slippage; targets are limit orders
    (maker fee, no slippage).
  * Funding is charged as a cost at every 8-hour settlement a trade spans.
One position at a time. Position size = risk % of current equity / stop
distance, capped by max leverage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    maker_fee: float = 0.0        # MEXC USDT-M futures (checked 2026): 0% maker
    taker_fee: float = 0.0002     # 0.02% taker
    slippage: float = 0.0001      # 0.01% on market/stop fills
    funding_per_8h: float = 0.0001  # charged as a cost each settlement held


@dataclass(frozen=True)
class RunConfig:
    target_r: float = 2.0
    max_hold_bars: int = 32
    limit_expiry_bars: int = 6
    risk_pct: float = 0.01
    start_equity: float = 10_000.0
    max_leverage: float = 10.0
    min_stop_atr: float = 0.25    # skip trades whose real stop is absurdly tight
    costs: Costs = Costs()


FUNDING_PERIOD = 8 * 3600


def run_backtest(df: pd.DataFrame, sig: pd.DataFrame, cfg: RunConfig, bar_seconds: int,
                 context: pd.DataFrame | None = None) -> tuple[list[dict], dict]:
    """df: candles (timestamp, open, high, low, close). sig: output of setups.signals.
    context: optional per-bar labels copied onto each trade (e.g. regime)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = ((df["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).to_numpy("int64")
    side_a = sig["side"].to_numpy(int)
    etype = sig["entry_type"].to_numpy(object)
    eprice = sig["entry"].to_numpy(float)
    stop_a = sig["stop"].to_numpy(float)
    atr_a = sig["atr"].to_numpy(float)
    cost = cfg.costs
    n = len(df)

    equity = cfg.start_equity
    trades: list[dict] = []
    stats = {"signals": 0, "limit_orders": 0, "limit_filled": 0, "limit_expired": 0, "skipped": 0}
    pending = None   # dict
    pos = None       # dict

    def close(i: int, price: float, reason: str, taker: bool):
        nonlocal equity, pos
        p = pos
        fee = cost.taker_fee if taker else cost.maker_fee
        if taker:
            price *= (1 - p["side"] * cost.slippage)
        gross = p["side"] * (price - p["entry"]) * p["qty"]
        exit_time = ts[i] + (bar_seconds if reason == "timeout" else 0)
        crossings = max(0, exit_time // FUNDING_PERIOD - p["time"] // FUNDING_PERIOD)
        notional = p["entry"] * p["qty"]
        costs = p["entry_fee"] + price * p["qty"] * fee + crossings * cost.funding_per_8h * notional
        pnl = gross - costs
        equity += pnl
        t = {
            "entry_time": pd.Timestamp(p["time"], unit="s", tz="UTC"),
            "exit_time": pd.Timestamp(exit_time, unit="s", tz="UTC"),
            "side": "long" if p["side"] == 1 else "short",
            "entry_type": p["etype"], "entry": p["entry"], "stop": p["stop"], "target": p["tp"],
            "exit": price, "exit_reason": reason, "bars_held": i - p["bar"] + (1 if reason == "timeout" else 0),
            "qty": p["qty"], "risk_usd": p["risk_usd"], "costs_usd": costs, "pnl_usd": pnl,
            "r": pnl / p["risk_usd"], "equity_after": equity,
        }
        if context is not None:
            t.update(context.iloc[p["signal_bar"]].to_dict())
        trades.append(t)
        pos = None

    def open_pos(i: int, side: int, entry: float, stop: float, etype_: str, a: float, signal_bar: int) -> bool:
        nonlocal pos
        risk_unit = (entry - stop) * side
        if not np.isfinite(risk_unit) or risk_unit <= 0 or (np.isfinite(a) and risk_unit < cfg.min_stop_atr * a):
            stats["skipped"] += 1
            return False
        risk_usd = equity * cfg.risk_pct
        qty = risk_usd / risk_unit
        max_qty = equity * cfg.max_leverage / entry
        if qty > max_qty:
            qty = max_qty
            risk_usd = qty * risk_unit
        fee = cost.maker_fee if etype_ == "limit" else cost.taker_fee
        pos = {"side": side, "entry": entry, "stop": stop, "tp": entry + side * cfg.target_r * risk_unit,
               "qty": qty, "risk_usd": risk_usd, "entry_fee": entry * qty * fee, "time": int(ts[i]),
               "bar": i, "etype": etype_, "signal_bar": signal_bar}
        return True

    for i in range(n):
        # 1) manage an open position (opened on an earlier bar)
        if pos is not None and pos["bar"] < i:
            s, st, tp = pos["side"], pos["stop"], pos["tp"]
            if s == 1:
                if o[i] <= st:
                    close(i, o[i], "stop", True)
                elif l[i] <= st:
                    close(i, st, "stop", True)
                elif h[i] >= tp:
                    close(i, max(o[i], tp), "target", False)
            else:
                if o[i] >= st:
                    close(i, o[i], "stop", True)
                elif h[i] >= st:
                    close(i, st, "stop", True)
                elif l[i] <= tp:
                    close(i, min(o[i], tp), "target", False)
            if pos is not None and i - pos["bar"] + 1 >= cfg.max_hold_bars:
                close(i, c[i], "timeout", True)

        # 2) work a pending entry order
        if pending is not None and pos is None:
            s = pending["side"]
            if pending["type"] == "market":
                entry = o[i] * (1 + s * cost.slippage)
                pending_ = pending
                pending = None
                if open_pos(i, s, entry, pending_["stop"], "market", pending_["atr"], pending_["bar"]):
                    st, tp = pos["stop"], pos["tp"]
                    if (s == 1 and l[i] <= st) or (s == -1 and h[i] >= st):
                        close(i, st, "stop", True)          # stop first if both possible
                    elif (s == 1 and h[i] >= tp) or (s == -1 and l[i] <= tp):
                        close(i, tp, "target", False)
            else:
                lim = pending["price"]
                touched = l[i] <= lim if s == 1 else h[i] >= lim
                if touched:
                    entry = min(o[i], lim) if s == 1 else max(o[i], lim)
                    pending_ = pending
                    pending = None
                    if open_pos(i, s, entry, pending_["stop"], "limit", pending_["atr"], pending_["bar"]):
                        stats["limit_filled"] += 1
                        st = pos["stop"]
                        if (s == 1 and l[i] <= st) or (s == -1 and h[i] >= st):
                            close(i, st, "stop", True)
                elif i >= pending["expires"]:
                    stats["limit_expired"] += 1
                    pending = None

        # 3) new setup at this bar's close (only when flat with nothing pending)
        if pos is None and pending is None and side_a[i] != 0 and i + 1 < n:
            stats["signals"] += 1
            if etype[i] == "limit":
                stats["limit_orders"] += 1
                pending = {"type": "limit", "side": int(side_a[i]), "price": eprice[i], "stop": stop_a[i],
                           "atr": atr_a[i], "bar": i, "expires": i + cfg.limit_expiry_bars}
            else:
                pending = {"type": "market", "side": int(side_a[i]), "stop": stop_a[i], "atr": atr_a[i], "bar": i}

    if pos is not None:   # still open at the end of the data: close at the last price
        close(n - 1, c[n - 1], "end_of_data", True)
    stats["final_equity"] = equity
    return trades, stats
