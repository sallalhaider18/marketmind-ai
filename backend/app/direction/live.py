"""Live PAPER trading with the saved direction models. No real orders, ever.

Same rules as the backtest (app/direction/pipeline.py):
  * At each 30-minute candle close in the New York session (candles opening
    9:30-15:30 ET, every day), the saved models estimate the chance of an up move.
  * Per coin per day, the first signal with confidence >= the threshold is
    traded: long above 50%, short below.
  * Entry at the current price, stop and target the same distance (BTC 1,000,
    ETH 100 by default), exit at target, stop or 9:29 ET the next day.
  * Stops and targets are checked against 1-minute candle highs and lows (so a
    spike between checks is not missed); if both were touched in the same
    minute, the stop counts.
Paper account: $10,000, 1% risk per trade, MEXC fees (0% maker on targets,
0.02% taker on entries, stops and time exits).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from app.direction.features import FEATURES, NY_LAST_BAR_MIN, NY_OPEN_MIN, build_features, sequences

NY = "America/New_York"
UK = "Europe/London"
TAKER, MAKER = 0.0002, 0.0


def fmt_times(t: datetime) -> str:
    ts = pd.Timestamp(t)
    return f"{ts.tz_convert(NY):%a %H:%M} NY / {ts.tz_convert(UK):%H:%M} UK"


class Market(Protocol):
    def candles_30m(self) -> dict[str, pd.DataFrame]: ...
    def funding(self, symbol: str) -> pd.DataFrame | None: ...
    def recent_range(self, symbol: str, since: datetime) -> tuple[float, float, float]: ...
    def last_price(self, symbol: str) -> float: ...


@dataclass
class PaperTrade:
    symbol: str
    side: str
    ny_date: str
    entry_time: str
    entry: float
    stop: float
    target: float
    qty: float
    risk_usd: float
    deadline: str          # 9:29 ET next day (ISO, UTC)
    p_up: float
    model: str
    checked_until: str     # last time stops/targets were checked (ISO, UTC)


@dataclass
class State:
    equity: float = 10_000.0
    open: dict = field(default_factory=dict)            # symbol -> PaperTrade dict
    traded_on: dict = field(default_factory=dict)       # symbol -> NY date of last trade
    last_decision: dict = field(default_factory=dict)   # symbol -> ISO open time of last decided candle
    closed: int = 0
    wins: int = 0
    total_r: float = 0.0


class Predictor:
    """Loads models/direction/{meta.json, ml.joblib, dl.pt}."""

    def __init__(self, model_dir: Path, model: str | None = None):
        import joblib

        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("No trained models found. Run first:  python -m scripts.direction_ml")
        self.meta = json.loads(meta_path.read_text())
        self.ml = joblib.load(model_dir / "ml.joblib")
        self.dl = None
        if "dl" in self.meta["models"] and (model_dir / "dl.pt").exists():
            try:
                from app.direction.models import LSTMModel
                self.dl = LSTMModel.from_bytes((model_dir / "dl.pt").read_bytes())
            except Exception as exc:   # PyTorch missing or incompatible
                print(f"Deep-learning model not loaded ({exc}); using the ML model only.")
        wanted = model or self.meta["live_model"]
        if wanted in ("dl", "ensemble") and self.dl is None:
            wanted = "ml"
        self.model = wanted
        self.threshold = float(self.meta["thresholds"].get(wanted, 0.1))
        self.distance = {k: float(v) for k, v in self.meta["distance"].items()}
        self.proven = bool(self.meta.get("proven", False))

    def __call__(self, X: pd.DataFrame, seq: np.ndarray) -> dict[str, float]:
        out = {"ml": float(self.ml.predict(X, seq)[0])}
        if self.dl is not None:
            out["dl"] = float(self.dl.predict(X, seq)[0])
            out["ensemble"] = (out["ml"] + out["dl"]) / 2
        return out


class PaperTrader:
    def __init__(self, market: Market, predictor: Callable, model: str, threshold: float,
                 distance: dict[str, float], state_path: Path, log_path: Path,
                 risk_pct: float = 0.01, out: Callable[[str], None] = print):
        self.m, self.predict, self.model, self.thr = market, predictor, model, threshold
        self.distance, self.state_path, self.log_path, self.risk_pct = distance, state_path, log_path, risk_pct
        self.say = out
        self.state = self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> State:
        if self.state_path.exists():
            d = json.loads(self.state_path.read_text())
            return State(**d)
        return State()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def _log(self, row: dict) -> None:
        new = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # -- one cycle -----------------------------------------------------------
    def step(self, now: datetime) -> None:
        self._manage_open(now)
        self._decide(now)
        self._save()

    @staticmethod
    def last_closed_bar(now: datetime) -> pd.Timestamp:
        """Open time of the most recent fully closed 30-minute candle (5 s grace for data)."""
        t = pd.Timestamp(now) - pd.Timedelta(seconds=5)
        return t.floor("30min") - pd.Timedelta(minutes=30)

    def _decide(self, now: datetime) -> None:
        bar = self.last_closed_bar(now)
        ny = bar.tz_convert(NY)
        mins = ny.hour * 60 + ny.minute
        if not (NY_OPEN_MIN <= mins <= NY_LAST_BAR_MIN):
            return
        todo = [s for s in self.distance if self.state.last_decision.get(s) != bar.isoformat()]
        if not todo:
            return
        candles = self.m.candles_30m()
        for sym in todo:
            df = candles.get(sym)
            if df is None or df.empty or df["timestamp"].iloc[-1] < bar:
                continue   # candle not available yet: try again next cycle
            df = df.sort_values("timestamp").reset_index(drop=True)
            other = next((candles[o] for o in candles if o != sym), None)
            f = build_features(df, sym, other, self.m.funding(sym))
            hits = np.flatnonzero((df["timestamp"] == bar).to_numpy())
            if len(hits) == 0:
                continue
            row = int(hits[-1])
            probs = self.predict(f[FEATURES].iloc[[row]], sequences(f, np.array([row])))
            p = probs[self.model]
            conf = abs(p - 0.5)
            ny_date = str(ny.date())
            others = " | ".join(f"{k.upper() if k != 'ensemble' else 'combined'} {v:.0%}" for k, v in probs.items())
            need = f"needs ≥{0.5 + self.thr:.0%} or ≤{0.5 - self.thr:.0%}"
            when = fmt_times(bar + pd.Timedelta(minutes=30))
            self.state.last_decision[sym] = bar.isoformat()
            if sym in self.state.open:
                self.say(f"{when}  {sym:<10} up chance {others}  -> trade already open")
            elif self.state.traded_on.get(sym) == ny_date:
                self.say(f"{when}  {sym:<10} up chance {others}  -> already traded today")
            elif conf >= self.thr and conf > 0:
                self._open(sym, "long" if p > 0.5 else "short", p, ny_date, now)
            else:
                self.say(f"{when}  {sym:<10} up chance {others}  -> no trade ({need})")

    def _open(self, sym: str, side: str, p: float, ny_date: str, now: datetime) -> None:
        price = self.m.last_price(sym)
        s = 1 if side == "long" else -1
        d = self.distance[sym]
        risk = self.state.equity * self.risk_pct
        qty = risk / d
        nd = pd.Timestamp(ny_date) + pd.Timedelta(days=1)
        deadline = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day, hour=9, minute=29, tz=NY)
        t = PaperTrade(sym, side, ny_date, pd.Timestamp(now).isoformat(), price, price - s * d, price + s * d,
                       qty, risk, deadline.tz_convert("UTC").isoformat(), p, self.model,
                       pd.Timestamp(now).isoformat())
        self.state.open[sym] = asdict(t)
        self.state.traded_on[sym] = ny_date
        arrow = "LONG  (buy)" if side == "long" else "SHORT (sell)"
        self.say(f"{fmt_times(now)}  {sym:<10} >>> PAPER {arrow} at {price:,.2f}   "
                 f"stop {t.stop:,.2f}   target {t.target:,.2f}   up chance {p:.0%}   risk ${risk:,.0f}")

    def _manage_open(self, now: datetime) -> None:
        for sym, td in list(self.state.open.items()):
            t = PaperTrade(**td)
            s = 1 if t.side == "long" else -1
            since = pd.Timestamp(t.checked_until).to_pydatetime()
            hi, lo, last = self.m.recent_range(sym, since)
            stop_hit = (lo <= t.stop) if s == 1 else (hi >= t.stop)
            tgt_hit = (hi >= t.target) if s == 1 else (lo <= t.target)
            if stop_hit:
                self._close(t, t.stop, "stop", now, taker=True)
            elif tgt_hit:
                self._close(t, t.target, "target", now, taker=False)
            elif pd.Timestamp(now) >= pd.Timestamp(t.deadline):
                self._close(t, last, "next-day close (9:29 ET)", now, taker=True)
            else:
                td["checked_until"] = (pd.Timestamp(now) - pd.Timedelta(minutes=1)).isoformat()

    def _close(self, t: PaperTrade, price: float, reason: str, now: datetime, taker: bool) -> None:
        s = 1 if t.side == "long" else -1
        fees = t.entry * t.qty * TAKER + price * t.qty * (TAKER if taker else MAKER)
        pnl = s * (price - t.entry) * t.qty - fees
        r = pnl / t.risk_usd
        st = self.state
        st.equity += pnl
        st.closed += 1
        st.wins += int(pnl > 0)
        st.total_r += r
        del st.open[t.symbol]
        word = "WIN " if pnl > 0 else "LOSS"
        self.say(f"{fmt_times(now)}  {t.symbol:<10} <<< CLOSED {t.side.upper()} at {price:,.2f} ({reason})   "
                 f"{word} {r:+.2f} R  ${pnl:+,.2f}   paper equity ${st.equity:,.2f}   "
                 f"[{st.wins}/{st.closed} wins, total {st.total_r:+.2f} R]")
        self._log({"symbol": t.symbol, "side": t.side, "ny_date": t.ny_date, "entry_time": t.entry_time,
                   "entry": t.entry, "stop": t.stop, "target": t.target, "exit_time": pd.Timestamp(now).isoformat(),
                   "exit": price, "reason": reason, "r": round(r, 4), "pnl_usd": round(pnl, 2),
                   "equity": round(st.equity, 2), "p_up": round(t.p_up, 4), "model": t.model})

    # -- status line ---------------------------------------------------------
    def status(self, now: datetime) -> str:
        st = self.state
        nxt = self.next_decision(now)
        opened = ", ".join(f"{s} {d['side']} from {d['entry']:,.2f}" for s, d in st.open.items()) or "none"
        rate = f"{st.wins}/{st.closed} wins" if st.closed else "no closed trades yet"
        return (f"{fmt_times(now)}  waiting. Next decision: {fmt_times(nxt)}. Open: {opened}. "
                f"Paper equity ${st.equity:,.2f} ({rate}, total {st.total_r:+.2f} R)")

    @staticmethod
    def next_decision(now: datetime) -> pd.Timestamp:
        t = pd.Timestamp(now).ceil("30min")
        for _ in range(200):
            ny = (t - pd.Timedelta(minutes=30)).tz_convert(NY)     # the candle that closes at t
            if NY_OPEN_MIN <= ny.hour * 60 + ny.minute <= NY_LAST_BAR_MIN:
                return t
            t += pd.Timedelta(minutes=30)
        return t


class MexcMarket:
    """Live data from MEXC (public endpoints only): 30m candles stored on disk and
    kept up to date, funding rates, and recent 1-minute highs/lows."""

    def __init__(self, adapter, symbols: list[str], raw_dir: Path, history_days: int = 400):
        self.a, self.symbols, self.raw_dir, self.days = adapter, symbols, raw_dir, history_days
        self._funding: dict[str, tuple[datetime, pd.DataFrame | None]] = {}

    def candles_30m(self) -> dict[str, pd.DataFrame]:
        from app.data.historical import update_history
        return {s: update_history(self.a, s, "30m", self.raw_dir, days=self.days)[1] for s in self.symbols}

    def funding(self, symbol: str) -> pd.DataFrame | None:
        now = datetime.now(timezone.utc)
        cached = self._funding.get(symbol)
        if cached and now - cached[0] < timedelta(hours=1):
            return cached[1]
        try:
            f = self.a.fetch_funding_history(symbol, now - timedelta(days=3), max_pages=2)
        except Exception:
            f = cached[1] if cached else None
        self._funding[symbol] = (now, f)
        return f

    def recent_range(self, symbol: str, since: datetime) -> tuple[float, float, float]:
        now = datetime.now(timezone.utc)
        df = self.a.fetch_candles(symbol, "1m", since, now + timedelta(minutes=1))
        last = self.a.fetch_last_price(symbol)
        if df.empty:
            return last, last, last
        return max(df["high"].max(), last), min(df["low"].min(), last), last

    def last_price(self, symbol: str) -> float:
        return self.a.fetch_last_price(symbol)
