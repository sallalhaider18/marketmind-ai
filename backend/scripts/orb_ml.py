"""Train and test an ML filter on top of the New York opening-range breakout.

Run from the backend/ folder:
    python -m scripts.orb_ml                 # downloads 3 years of 5m + 15m candles and funding
    python -m scripts.orb_ml --no-download

Base trades: entry on the breakout close, stop at the opposite side of the range,
targets 1x / 2x / partial / 1:1 / 1.5:1 / 2:1, on 5m and 15m candles (12 configurations), with BTC and
ETH pooled. For each configuration:
  * Development = first 70% of trades in time; Test = last 30%.
  * Out-of-fold predictions on development (always trained on earlier trades only)
    choose the cut-off; a final model trained on development predicts Test.
  * Eligible = filtered development (out-of-fold) result >= 30 trades, average > 0
    and t-stat >= 2; filtered Test >= 10 trades and average > 0.
Writes reports/orb_ml_<date>.md, reports/orb_ml_test_trades_<date>.csv and
models/orb_filter_<tf>_<target>.joblib.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.backtest.metrics import summarize
from app.config import PROJECT_ROOT, get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.logging_setup import setup_logging
from app.ml.orb_filter import auc, choose_threshold, make_model, oof_predictions, permutation_importance_auc
from app.strategy.orb import TARGETS, OrbConfig, run_orb
from app.strategy.orb_features import FEATURES, orb_features
from scripts.compare_strategies import DEV_SHARE, MIN_DEV_T, MIN_DEV_TRADES, MIN_TEST_TRADES, num, pct

TIMEFRAMES = {"5m": 5, "15m": 15}


def fixed_risk(t: pd.DataFrame) -> pd.DataFrame:
    """Re-express results with a fixed $100 risk per trade so filtered and
    unfiltered sets compare fairly (no compounding effects)."""
    t = t.copy()
    t["costs_usd"] = t["costs_usd"] / t["risk_usd"] * 100
    t["pnl_usd"] = t["r"] * 100
    t["risk_usd"] = 100.0
    return t


def load_funding(adapter, sym, raw_dir, since) -> pd.DataFrame | None:
    path = raw_dir / "mexc_futures" / sym.upper() / "funding.parquet"
    if adapter:
        try:
            f = adapter.fetch_funding_history(sym, since)
            if len(f):
                path.parent.mkdir(parents=True, exist_ok=True)
                f.to_parquet(path, index=False)
        except Exception as exc:
            print(f"  funding history download failed ({type(exc).__name__}: {exc}); continuing without it")
    if path.exists():
        return pd.read_parquet(path)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES), choices=list(TIMEFRAMES))
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--years", type=float, default=3.0)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    symbols = args.symbols or s.symbols
    days = int(args.years * 365)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    adapter = None if args.no_download else MexcFuturesAdapter(s.mexc_rest_url)

    candles, funding, notes = {}, {}, []
    try:
        for sym in symbols:
            funding[sym] = load_funding(adapter, sym, s.raw_dir, since.to_pydatetime())
            notes.append(f"Funding {sym}: " + (f"{len(funding[sym])} settlements" if funding[sym] is not None
                                               else "not available (feature left empty)"))
            for tf in args.timeframes:
                if adapter:
                    print(f"Updating {sym} {tf} history ({days} days)...", flush=True)
                    try:
                        update_history(adapter, sym, tf, s.raw_dir, days=days)
                    except Exception as exc:
                        print(f"  download failed ({type(exc).__name__}: {exc}); using what is on disk")
                df = load_candles(candle_path(s.raw_dir, "mexc_futures", sym, tf))
                df = df[df["timestamp"] >= since].reset_index(drop=True)
                candles[(sym, tf)] = df
                if len(df):
                    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
                    notes.append(f"{tf} candles {sym}: {len(df):,} over {span} days")
    finally:
        if adapter:
            adapter.close()

    results, test_rows = [], []
    out_models = PROJECT_ROOT / "models"
    out_models.mkdir(exist_ok=True)
    for tf in args.timeframes:
        for target in TARGETS:
            parts = []
            for sym in symbols:
                df = candles.get((sym, tf))
                if df is None or len(df) < 1000:
                    continue
                trades, _ = run_orb(df, TIMEFRAMES[tf], OrbConfig(entry="close", stop="range", target=target))
                t = pd.DataFrame(trades)
                if t.empty:
                    continue
                X = orb_features(df, TIMEFRAMES[tf], t, sym, funding.get(sym))
                parts.append(pd.concat([fixed_risk(t).assign(symbol=sym), X], axis=1))
            if not parts:
                continue
            data = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
            res = evaluate(data, tf, target, out_models)
            print(f"{tf:>4} {target:<8} base test exp {num(res['base_test']['expectancy_r'])} R | "
                  f"filtered test exp {num(res['filt_test']['expectancy_r'])} R "
                  f"({res['filt_test']['trades']} of {res['base_test']['trades']} trades) | "
                  f"{'ELIGIBLE' if res['eligible'] else 'not eligible'}", flush=True)
            results.append(res)
            test_rows.append(res.pop("test_trades"))

    if not results:
        print("No results: no data available.")
        return
    write_report(results, test_rows, notes)


def evaluate(data: pd.DataFrame, tf: str, target: str, out_models) -> dict:
    cut = data["entry_time"].iloc[int(len(data) * DEV_SHARE)]
    dev, test = data[data["entry_time"] < cut].reset_index(drop=True), data[data["entry_time"] >= cut]
    y_dev = (dev["r"] > 0).to_numpy(int)
    y_test = (test["r"] > 0).to_numpy(int)

    p_oof = oof_predictions(dev[FEATURES], y_dev)
    has = ~np.isnan(p_oof)
    min_keep = max(40, int(0.25 * has.sum()))
    thr, thr_info = choose_threshold(p_oof, dev["r"].to_numpy(), min_keep)

    model = make_model().fit(dev[FEATURES], y_dev)
    p_test = model.predict_proba(test[FEATURES])[:, 1]

    oof_all = dev[has]
    oof_kept = dev[has & (p_oof >= thr)]
    test_kept = test[p_test >= thr]
    base_oof, filt_oof = summarize(oof_all), summarize(oof_kept)
    base_test, filt_test = summarize(test), summarize(test_kept)
    eligible = (filt_oof["trades"] >= MIN_DEV_TRADES and (filt_oof["expectancy_r"] or 0) > 0
                and (filt_oof["t_stat"] or 0) >= MIN_DEV_T
                and filt_test["trades"] >= MIN_TEST_TRADES and (filt_test["expectancy_r"] or 0) > 0)

    importance = permutation_importance_auc(model, test[FEATURES], y_test)[:8]

    per_symbol = {}
    for sym, g in test.assign(p=p_test).groupby("symbol"):
        per_symbol[sym] = {"base": summarize(g), "filtered": summarize(g[g["p"] >= thr])}

    joblib.dump({"model": model, "features": FEATURES, "threshold": thr, "timeframe": tf, "target": target,
                 "trained_until": str(cut), "base": "ORB close entry, range stop"},
                out_models / f"orb_filter_{tf}_{target}.joblib")

    return {
        "tf": tf, "target": target, "threshold": thr, "thr_info": thr_info, "cut": cut,
        "base_oof": base_oof, "filt_oof": filt_oof, "base_test": base_test, "filt_test": filt_test,
        "auc_oof": auc(y_dev[has], p_oof[has]), "auc_test": auc(y_test, p_test),
        "eligible": eligible, "importance": importance, "per_symbol": per_symbol,
        "test_trades": test.assign(p_win=p_test, taken=p_test >= thr, timeframe=tf, target_cfg=target)[
            ["entry_time", "symbol", "side", "timeframe", "target_cfg", "r", "exit_reason", "p_win", "taken"]],
    }


def fmt_auc(a) -> str:
    return "—" if a is None else f"{a:.2f}"


def write_report(results, test_rows, notes) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.concat(test_rows, ignore_index=True).to_csv(out / f"orb_ml_test_trades_{stamp}.csv", index=False)

    eligible = sorted([r for r in results if r["eligible"]],
                      key=lambda r: r["filt_oof"]["win_rate"] or 0, reverse=True)
    L = [f"# ORB + ML filter, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
         "Base: New York opening-range breakout, entry on the breakout close, stop at the opposite side of "
         "the range. BTC and ETH pooled. The filter skips breakouts whose predicted chance of ending in profit "
         "is below a cut-off chosen on development data only. All results after fees, slippage and funding, "
         "with a fixed 1R risk per trade.", "",
         "*Base* = every breakout. *Filtered* = only those the model takes. *Dev (OOF)* = development trades "
         "predicted by models trained only on earlier trades. *AUC* measures how well the model ranks winners "
         "above losers: 0.50 = no skill. With only a few hundred test trades, values between about 0.40 and "
         "0.60 can happen by pure chance, so judge the filter by the eligibility rule, not by AUC alone.", ""]
    if eligible:
        w = eligible[0]
        L += ["## Result", "", f"**Best eligible: {w['tf']} candles, target {w['target']}.** Test period: "
              f"{w['filt_test']['trades']} of {w['base_test']['trades']} breakouts taken, win rate "
              f"{pct(w['filt_test']['win_rate'])} (all breakouts: {pct(w['base_test']['win_rate'])}), "
              f"average {num(w['filt_test']['expectancy_r'])} R per trade (all breakouts: "
              f"{num(w['base_test']['expectancy_r'])} R).", ""]
        if (w["filt_test"]["win_rate"] or 0) < 0.6:
            L += [f"Win rate on the test period is below the 60% goal.", ""]
    else:
        L += ["## Result", "", "**No configuration is eligible.** The filter did not produce a result that is "
              "both clearly profitable on development data (t-stat >= 2) and profitable on the unseen test "
              "period.", ""]

    L += ["## All configurations", "",
          "| Candles | Target | Test: all | Test: filtered | Test win all → filtered | Dev (OOF) filtered | "
          "Dev t | AUC dev / test | Eligible |", "|---|---|---|---|---|---|---:|---|:---:|"]
    for r in results:
        bt, ft, fo = r["base_test"], r["filt_test"], r["filt_oof"]
        L.append(f"| {r['tf']} | {r['target']} | {bt['trades']} tr, {num(bt['expectancy_r'])} R | "
                 f"{ft['trades']} tr, {num(ft['expectancy_r'])} R | {pct(bt['win_rate'])} → {pct(ft['win_rate'])} | "
                 f"{fo['trades']} tr, {num(fo['expectancy_r'])} R | {num(fo['t_stat'], 1)} | "
                 f"{fmt_auc(r['auc_oof'])} / {fmt_auc(r['auc_test'])} | "
                 f"{'yes' if r['eligible'] else ''} |")

    for r in results:
        L += ["", f"### {r['tf']} candles, target {r['target']}", "",
              f"Cut-off: take a breakout if predicted win chance ≥ {r['threshold']:.2f} "
              f"(on development, kept {r['thr_info']['kept']} of {r['thr_info']['of']}). Test period starts "
              f"{r['cut']:%Y-%m-%d}.", ""]
        for sym, v in r["per_symbol"].items():
            L.append(f"- {sym} test: all {v['base']['trades']} trades, {pct(v['base']['win_rate'])} win, "
                     f"{num(v['base']['expectancy_r'])} R → filtered {v['filtered']['trades']} trades, "
                     f"{pct(v['filtered']['win_rate'])} win, {num(v['filtered']['expectancy_r'])} R")
        if r["importance"]:
            L += ["", "Most useful features on the test period (drop in AUC when shuffled):", ""]
            L += [f"- {name}: {val:+.3f}" for name, val in r["importance"]]

    L += ["", "## Data", ""] + [f"- {n}" for n in notes]
    L += ["", f"{len(results)} configurations were tried, so one may look good by chance; the eligibility rule and the "
          "untouched test period are there to catch that. Past results do not guarantee future results."]
    path = out / f"orb_ml_{stamp}.md"
    path.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 96)
    print(f"{'Candles':<8}{'Target':<9}{'Test all':>18}{'Test filtered':>20}{'Win all→filt':>17}{'Dev t':>7}"
          f"{'AUC test':>10}  Eligible")
    for r in results:
        bt, ft = r["base_test"], r["filt_test"]
        print(f"{r['tf']:<8}{r['target']:<9}{bt['trades']:>6} {num(bt['expectancy_r']):>6} R  "
              f"{ft['trades']:>8} {num(ft['expectancy_r']):>6} R  {pct(bt['win_rate']):>7}→{pct(ft['win_rate']):<7}"
              f"{num(r['filt_oof']['t_stat'], 1):>7}{fmt_auc(r['auc_test']):>10}  "
              f"{'YES' if r['eligible'] else ''}")
    print("=" * 96)
    if eligible:
        w = eligible[0]
        print(f"BEST ELIGIBLE: {w['tf']} candles, target {w['target']}: test win rate "
              f"{pct(w['filt_test']['win_rate'])}, {num(w['filt_test']['expectancy_r'])} R per trade")
    else:
        print("No configuration is eligible. See the report for what the model learned.")
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
