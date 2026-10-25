"""Train and honestly test the ML and deep-learning direction models, then save
them for live paper trading.

Run from the backend/ folder:
    python -m scripts.direction_ml                  # downloads 30m history (up to 3 years) + funding
    python -m scripts.direction_ml --no-download
    python -m scripts.direction_ml --btc 1000 --eth 100   # stop = target distance (1:1)
    python -m scripts.direction_ml --no-dl          # skip the deep-learning model

What it does:
  1. Builds one example per 30-minute candle close in the New York session
     (9:30-16:00 ET, every day) with 30m / 4h / daily features, and what a 1:1 trade
     (BTC +/-1,000, ETH +/-100) would have done from there.
  2. First 70% of history = development, last 30% = test (never used for choices).
     On development, models are trained on earlier data and predict later data;
     the confidence threshold is chosen on those predictions.
  3. Models trained on all development data then trade the test period with the
     daily rule (first confident signal per coin per day). Compared with always-long,
     always-short and random direction on the same days.
  4. Final models are retrained on ALL data and saved to models/direction/ for
     scripts.live_trader.
Eligible = development result >= 30 trades, average > 0 and t-stat >= 2, AND test
result >= 10 trades with average > 0.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.config import PROJECT_ROOT, get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.direction.features import FEATURES, SEQ_LEN
from app.direction.models import TORCH, make, model_names
from app.direction.pipeline import (baselines, build_dataset, choose_threshold, fit_predict, oof_predictions,
                                    r_stats, select_trades, train_rows)
from app.logging_setup import setup_logging
from scripts.orb_ml import load_funding

DEV_SHARE = 0.70
MIN_DEV_TRADES, MIN_TEST_TRADES, MIN_T = 30, 10, 2.0
LABELS = {"ml": "ML (gradient-boosted trees)", "dl": "Deep learning (LSTM)", "ensemble": "ML + DL combined"}
MODEL_DIR = PROJECT_ROOT / "models" / "direction"


def pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def sgn(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{d}f}"


def auc(y, p):
    ok = ~np.isnan(y) & ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok])) if ok.sum() > 20 and len(np.unique(y[ok])) == 2 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--btc", type=float, default=1000.0, help="BTC stop and target distance (USDT)")
    ap.add_argument("--eth", type=float, default=100.0, help="ETH stop and target distance (USDT)")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-dl", action="store_true", help="skip the deep-learning model")
    args = ap.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    symbols = s.symbols
    dist = {sym: (args.btc if sym.upper().startswith("BTC") else args.eth) for sym in symbols}
    days = int(args.years * 365)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    if not args.no_dl and not TORCH:
        print("PyTorch is not installed, so the deep-learning model is skipped. Install it with:\n"
              "    pip install -r requirements.txt\n")

    adapter = None if args.no_download else MexcFuturesAdapter(s.mexc_rest_url)
    candles, funding, notes = {}, {}, []
    try:
        for sym in symbols:
            funding[sym] = load_funding(adapter, sym, s.raw_dir, since.to_pydatetime())
            if adapter:
                print(f"Updating {sym} 30m history (up to {days} days)...", flush=True)
                try:
                    update_history(adapter, sym, "30m", s.raw_dir, days=days)
                except Exception as exc:
                    print(f"  download failed ({type(exc).__name__}: {exc}); using what is on disk")
            df = load_candles(candle_path(s.raw_dir, "mexc_futures", sym, "30m"))
            df = df[df["timestamp"] >= since].reset_index(drop=True)
            if len(df) < 2000:
                print(f"  {sym}: only {len(df)} 30m candles, skipping")
                continue
            candles[sym] = df
            span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
            notes.append(f"{sym}: {len(df):,} 30m candles over {span} days "
                         f"({df['timestamp'].iloc[0]:%Y-%m-%d} to {df['timestamp'].iloc[-1]:%Y-%m-%d}); "
                         f"stop/target ±{dist[sym]:,.0f}")
            print(f"  {notes[-1]}", flush=True)
    finally:
        if adapter:
            adapter.close()
    if not candles:
        print("No data. Run without --no-download first.")
        return

    print("Building features and outcomes...", flush=True)
    data, seq = build_dataset(candles, funding, dist)
    names = model_names(use_dl=not args.no_dl)
    cut = int(len(data) * DEV_SHARE)
    dev, test = data.iloc[:cut].reset_index(drop=True), data.iloc[cut:].reset_index(drop=True)
    seq_dev, seq_test = seq[:cut], seq[cut:]
    split_time = test["decision_time"].iloc[0]
    print(f"{len(data):,} decisions; development until {split_time:%Y-%m-%d}, test after.", flush=True)

    print(f"Training on earlier data, predicting later data ({', '.join(names)})...", flush=True)
    p_oof = oof_predictions(dev, seq_dev, names)
    lab_dev = train_rows(dev)
    print("Training on all development data, predicting the test period...", flush=True)
    p_test = fit_predict(names, dev[FEATURES][lab_dev], seq_dev[lab_dev], dev["up_first"][lab_dev].to_numpy(int),
                         test[FEATURES], seq_test)

    results = {}
    for n in names:
        thr, _ = choose_threshold(dev, p_oof[n], min_trades=40)
        dev_tr = select_trades(dev, p_oof[n], thr)
        test_tr = select_trades(test, p_test[n], thr)
        res = {"threshold": thr, "dev": r_stats(dev_tr), "test": r_stats(test_tr),
               "auc_dev": auc(dev["up_first"].to_numpy(float), p_oof[n]),
               "auc_test": auc(test["up_first"].to_numpy(float), p_test[n]),
               "per_symbol": {sym: {"dev": r_stats(dev_tr[dev_tr["symbol"] == sym]),
                                    "test": r_stats(test_tr[test_tr["symbol"] == sym])} for sym in candles},
               "test_trades": test_tr.assign(model=n)}
        res["eligible"] = _eligible(res["dev"], res["test"])
        for sym, v in res["per_symbol"].items():
            v["eligible"] = _eligible(v["dev"], v["test"])
        results[n] = res
    base = {k: r_stats(v) for k, v in baselines(test).items()}

    # Live model: an eligible one (best test result), otherwise the combined model, flagged unproven
    elig = [n for n in names if results[n]["eligible"]]
    live_model = max(elig, key=lambda n: results[n]["test"]["exp_r"]) if elig else names[-1]
    proven = bool(elig)

    print("Training final models on all data for live use...", flush=True)
    lab_all = train_rows(data)
    save_models(names, data[FEATURES][lab_all], seq[lab_all], data["up_first"][lab_all].to_numpy(int),
                {n: results[n]["threshold"] for n in names}, dist, live_model, proven, data, results)

    write_report(results, base, notes, names, dist, split_time, live_model, proven)


def _eligible(d, t) -> bool:
    return (d["trades"] >= MIN_DEV_TRADES and (d["exp_r"] or 0) > 0 and (d["t"] or 0) >= MIN_T
            and t["trades"] >= MIN_TEST_TRADES and (t["exp_r"] or 0) > 0)


def save_models(names, X, S, y, thresholds, dist, live_model, proven, data, results) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        if n == "ensemble":
            continue
        m = make(n).fit(X, S, y)
        if n == "ml":
            joblib.dump(m, MODEL_DIR / "ml.joblib")
        else:
            (MODEL_DIR / "dl.pt").write_bytes(m.to_bytes())
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "trained_until": str(data["decision_time"].iloc[-1]),
        "models": names, "thresholds": thresholds, "distance": dist, "features": FEATURES,
        "seq_len": SEQ_LEN, "live_model": live_model, "proven": proven,
        "test_summary": {n: {k: results[n]["test"][k] for k in ("trades", "win_rate", "exp_r")} for n in names},
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def write_report(results, base, notes, names, dist, split_time, live_model, proven) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.concat([results[n]["test_trades"] for n in names], ignore_index=True).to_csv(
        out / f"direction_test_trades_{stamp}.csv", index=False)

    L = [f"# Direction models (ML and deep learning), {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
         "Every 30-minute candle close in the New York session (9:30-16:00 ET, every day) the models estimate "
         "the chance that price goes up the stop/target distance before it goes down by it. Per coin per day, "
         "the first signal whose confidence reaches the threshold is traded: long above 50%, short below. "
         f"Stop = target (1:1): " + ", ".join(f"{k} ±{v:,.0f}" for k, v in dist.items()) + ". Entry at the next "
         "candle's open; exit at target, stop or 9:29 ET the next day. Results in R (1R = the stop distance) "
         "after fees, slippage and funding; a candle touching both stop and target counts as a loss. With 1:1, "
         "a win rate a little above 50% is needed to cover costs.", "",
         f"Test period: from {split_time:%Y-%m-%d} (last 30%), not used for any choice. *AUC* = how well the "
         "model ranks up-moves above down-moves (0.50 = no skill; with this little data, 0.45-0.55 is "
         "normal for pure chance).", ""]
    L += ["## Test period results", "", "| Model | Confidence ≥ | Trades | Win rate | Avg R | Total R | Worst drop R | "
          "Dev t | AUC test | Eligible |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for n in names:
        r = results[n]
        t, d = r["test"], r["dev"]
        L.append(f"| {LABELS[n]} | {50 + r['threshold'] * 100:.0f}% | {t['trades']} | {pct(t['win_rate'])} | "
                 f"{sgn(t['exp_r'])} | {sgn(t['total_r'], 1)} | {sgn(t['max_dd_r'], 1)} | {sgn(d['t'], 1)} | "
                 f"{'—' if r['auc_test'] is None else format(r['auc_test'], '.2f')} | "
                 f"{'yes' if r['eligible'] else ''} |")
    for k, b in base.items():
        L.append(f"| Baseline: {k} | — | {b['trades']} | {pct(b['win_rate'])} | {sgn(b['exp_r'])} | "
                 f"{sgn(b['total_r'], 1)} | {sgn(b['max_dd_r'], 1)} | — | — | |")
    L += ["", "## Per coin (test period)", "", "| Model | Coin | Trades | Win rate | Avg R | Target / stop / next-day close |",
          "|---|---|---:|---:|---:|---|"]
    for n in names:
        for sym, v in results[n]["per_symbol"].items():
            t = v["test"]
            L.append(f"| {LABELS[n]} | {sym} | {t['trades']} | {pct(t['win_rate'])} | {sgn(t['exp_r'])} | "
                     f"{pct(t['target'])} / {pct(t['stop'])} / {pct(t['time'])} |")
    L += ["", "## Live paper trading", "",
          f"`python -m scripts.live_trader` will use **{LABELS[live_model]}** with confidence ≥ "
          f"{50 + results[live_model]['threshold'] * 100:.0f}%. "
          + ("It passed the eligibility checks." if proven else
             "**No model passed the eligibility checks**, so live trades are an experiment: paper only, "
             "and treat the results as a test of the model, not as a strategy that works."), "",
          "## Data", ""] + [f"- {x}" for x in notes]
    L += ["", "Three models were compared, so one may look good by chance; the eligibility rule and the untouched "
          "test period are there to catch that. Past results do not guarantee future results."]
    path = out / f"direction_ml_{stamp}.md"
    path.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"{'Model':<30}{'Conf ≥':>7}{'Trades':>8}{'Win':>8}{'Avg R':>8}{'Total R':>9}{'Dev t':>7}{'AUC':>6}  Eligible")
    for n in names:
        r = results[n]
        t = r["test"]
        print(f"{LABELS[n]:<30}{50 + r['threshold'] * 100:>6.0f}%{t['trades']:>8}{pct(t['win_rate']):>8}"
              f"{sgn(t['exp_r']):>8}{sgn(t['total_r'], 1):>9}{sgn(r['dev']['t'], 1):>7}"
              f"{('—' if r['auc_test'] is None else format(r['auc_test'], '.2f')):>6}  {'YES' if r['eligible'] else ''}")
    for k, b in base.items():
        print(f"{'Baseline: ' + k:<30}{'—':>7}{b['trades']:>8}{pct(b['win_rate']):>8}{sgn(b['exp_r']):>8}"
              f"{sgn(b['total_r'], 1):>9}")
    print("=" * 100)
    for n in names:
        for sym, v in results[n]["per_symbol"].items():
            t = v["test"]
            print(f"  {LABELS[n]:<28} {sym:<10} {t['trades']:>4} trades, win {pct(t['win_rate'])}, avg "
                  f"{sgn(t['exp_r'])} R  (target {pct(t['target'])}, stop {pct(t['stop'])}, "
                  f"next-day close {pct(t['time'])})")
    print(f"\nLive model: {LABELS[live_model]}" + ("" if proven else "  (NOT proven: paper trading only, as an experiment)"))
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
