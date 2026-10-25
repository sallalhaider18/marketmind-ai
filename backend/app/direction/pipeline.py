"""Dataset, honest evaluation and trade selection for the direction models.

Trade rule (identical in the backtest and in live paper trading):
  * Decisions at each 30-minute candle close during the New York session
    (candles opening 9:30-15:30 ET), every day of the week.
  * The model gives p = chance that price goes +distance before -distance.
    Confidence = |p - 0.5|. Long if p > 0.5, short if p < 0.5.
  * Per coin, per day: the FIRST decision whose confidence reaches the
    threshold is traded; nothing else that day. ("The highest of the day"
    cannot be known until the day is over, so a high threshold plays that role.)
  * 1:1 stop and target, entry at the next candle's open, exit at target, stop,
    or 9:29 ET the next day.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.backtest.engine import Costs
from app.direction.features import FEATURES, build_features, sequences
from app.direction.labels import label_decisions
from app.direction.models import make

THRESHOLDS = np.round(np.arange(0.0, 0.301, 0.01), 2)


def build_dataset(candles: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame | None],
                  dist: dict[str, float], cost: Costs = Costs()) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows = decision candles of all symbols, sorted by time; seq aligned with rows."""
    parts, seqs = [], []
    syms = list(candles)
    for sym in syms:
        df = candles[sym].sort_values("timestamp").reset_index(drop=True)
        other_sym = next((s for s in syms if s != sym), None)
        other = candles[other_sym] if other_sym else None
        f = build_features(df, sym, other, funding.get(sym))
        dec = np.flatnonzero(f["is_decision"].to_numpy())
        if len(dec) == 0:
            continue
        lab = label_decisions(df, dec, f["ny_date"].iloc[dec], dist[sym], cost)
        rows = f.iloc[dec][FEATURES + ["decision_time", "ny_date"]].copy()
        rows = pd.concat([rows, lab], axis=1)
        rows["symbol"] = sym
        rows["price"] = df["close"].iloc[dec].to_numpy()
        keep = rows["long_r"].notna().to_numpy()       # drop the last day (no outcome yet)
        parts.append(rows[keep])
        seqs.append(sequences(f, dec[keep]))
    if not parts:
        return pd.DataFrame(), np.zeros((0, 1, 1), np.float32)
    data = pd.concat(parts, ignore_index=True)
    seq = np.concatenate(seqs)
    order = np.argsort(data["decision_time"].to_numpy(), kind="stable")
    return data.iloc[order].reset_index(drop=True), seq[order]


def fit_predict(names, X_tr, S_tr, y_tr, X_te, S_te) -> dict[str, np.ndarray]:
    out = {}
    for n in [m for m in names if m != "ensemble"]:
        out[n] = make(n).fit(X_tr, S_tr, y_tr).predict(X_te, S_te)
    if "ensemble" in names:
        out["ensemble"] = (out["ml"] + out["dl"]) / 2
    return out


def train_rows(data: pd.DataFrame) -> np.ndarray:
    return data["up_first"].notna().to_numpy()


def oof_predictions(data: pd.DataFrame, seq: np.ndarray, names, n_splits: int = 4) -> dict[str, np.ndarray]:
    """Predictions for each row from models trained only on EARLIER rows."""
    p = {n: np.full(len(data), np.nan) for n in names}
    lab = train_rows(data)
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(data):
        tr = tr[lab[tr]]
        if len(tr) < 200 or len(np.unique(data["up_first"].iloc[tr])) < 2:
            continue
        preds = fit_predict(names, data[FEATURES].iloc[tr], seq[tr], data["up_first"].iloc[tr].to_numpy(int),
                            data[FEATURES].iloc[te], seq[te])
        for n, v in preds.items():
            p[n][te] = v
    return p


def select_trades(data: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    """Apply the daily rule. Returns one row per trade."""
    d = data.assign(p=p, conf=np.abs(p - 0.5))
    d = d[d["p"].notna() & (d["conf"] >= thr) & (d["conf"] > 0)]
    first = d.sort_values("decision_time").groupby(["symbol", "ny_date"], sort=False).head(1)
    side = np.where(first["p"] > 0.5, 1, -1)
    r = np.where(side == 1, first["long_r"], first["short_r"])
    exit_reason = np.where(side == 1, first["long_exit"], first["short_exit"])
    return pd.DataFrame({
        "decision_time": first["decision_time"].to_numpy(), "symbol": first["symbol"].to_numpy(),
        "ny_date": first["ny_date"].to_numpy(), "side": np.where(side == 1, "long", "short"),
        "p_up": first["p"].to_numpy(), "price": first["price"].to_numpy(), "r": r, "exit": exit_reason,
    }).sort_values("decision_time").reset_index(drop=True)


def r_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "exp_r": None, "t": None, "total_r": 0.0, "max_dd_r": 0.0,
                "target": 0.0, "stop": 0.0, "time": 0.0}
    r = trades["r"].astype(float)
    eq = r.cumsum()
    dd = float((eq - eq.cummax().clip(lower=0)).min())
    sd = r.std(ddof=1) if n > 1 else 0.0
    ex = trades["exit"].value_counts(normalize=True)
    return {"trades": n, "win_rate": float((r > 0).mean()), "exp_r": float(r.mean()),
            "t": float(r.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else None,
            "total_r": float(r.sum()), "max_dd_r": min(0.0, dd),
            "target": float(ex.get("target", 0.0)), "stop": float(ex.get("stop", 0.0)),
            "time": float(ex.get("next-day close (9:29 ET)", 0.0))}


def choose_threshold(data: pd.DataFrame, p: np.ndarray, min_trades: int) -> tuple[float, dict]:
    """Confidence threshold with the best average result on these (out-of-fold)
    predictions, keeping at least `min_trades` trades."""
    best_thr, best = 0.0, None
    for thr in THRESHOLDS:
        tr = select_trades(data, p, thr)
        if len(tr) < min_trades:
            break
        s = r_stats(tr)
        if best is None or s["exp_r"] > best["exp_r"]:
            best_thr, best = float(thr), s
    return best_thr, (best or r_stats(select_trades(data, p, 0.0)))


def baselines(data: pd.DataFrame, seed: int = 0) -> dict[str, dict]:
    """Simple comparisons on the same days: always long / always short at the first
    decision (10:00 ET), and a random direction."""
    first = data.sort_values("decision_time").groupby(["symbol", "ny_date"], sort=False).head(1)
    rng = np.random.default_rng(seed)
    rnd = rng.choice([1, -1], len(first))

    def mk(r, ex):
        return pd.DataFrame({"r": r, "exit": ex, "symbol": first["symbol"].to_numpy()})
    return {
        "always long": mk(first["long_r"].to_numpy(), first["long_exit"].to_numpy()),
        "always short": mk(first["short_r"].to_numpy(), first["short_exit"].to_numpy()),
        "random": mk(np.where(rnd == 1, first["long_r"], first["short_r"]),
                     np.where(rnd == 1, first["long_exit"], first["short_exit"])),
    }
