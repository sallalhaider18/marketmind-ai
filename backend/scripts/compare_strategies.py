"""Test every trade length x setup x profit target x symbol and rank them.

Run from the backend/ folder:
    python -m scripts.compare_strategies               # downloads what's missing, then tests
    python -m scripts.compare_strategies --no-download # use only data already on disk
    python -m scripts.compare_strategies --profiles intraday swing

History used (downloaded automatically, then kept up to date):
    scalp     1m candles,  last 90 days
    intraday  15m candles, last 365 days
    swing     1h candles,  last 1095 days

How the winner is chosen (to avoid fooling ourselves):
  * Each history is split in time: the first 70% is the DEVELOPMENT period,
    the last 30% is the TEST period that plays no part in the ranking.
  * Eligible = at least 30 development trades and 10 test trades, an average
    result per trade above zero after all costs in BOTH periods, AND a
    development result at least 2 standard errors above zero (t-stat >= 2),
    so that luck across 36 combinations is unlikely to produce a "winner".
    (Checked: on pure random data the simpler rule passed 3 of 36.)
  * Among eligible combinations, the highest development win rate wins.
A high win rate alone is not enough: a tiny target and a wide stop can win
most trades and still lose money.

Reports are written to reports/strategy_comparison_<date>.md / .csv and
reports/strategy_trades_<date>.csv.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import pandas as pd

from app.adapters.base import INTERVAL_SECONDS
from app.adapters.mexc_futures import MexcFuturesAdapter
from app.backtest.engine import RunConfig, run_backtest
from app.backtest.metrics import by_group, summarize
from app.config import PROJECT_ROOT, get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.data.validation import validate_candles
from app.logging_setup import setup_logging
from app.strategy.profiles import PROFILES, SETUPS, TARGETS_R
from app.strategy.setups import features, signals

DEV_SHARE = 0.70
MIN_DEV_TRADES, MIN_TEST_TRADES = 30, 10
MIN_DEV_T = 2.0


def regime_context(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    trend = pd.Series("sideways", index=df.index)
    trend[f["trend_up"]] = "uptrend"
    trend[f["trend_down"]] = "downtrend"
    rel_vol = (f["atr"] / df["close"])
    pct = rel_vol.rolling(500, min_periods=100).rank(pct=True)
    vol = pd.Series("normal volatility", index=df.index)
    vol[pct >= 0.8] = "high volatility"
    return pd.DataFrame({"trend_regime": trend, "vol_regime": vol})


def slice_start_equity(trades: pd.DataFrame, default: float) -> float:
    if trades.empty:
        return default
    first = trades.iloc[0]
    return first["equity_after"] - first["pnl_usd"]


def pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:+.{d}f}" if d else f"{x:.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    symbols = args.symbols or s.symbols
    adapter = None if args.no_download else MexcFuturesAdapter(s.mexc_rest_url)

    rows, all_trades, data_notes = [], [], []
    try:
        for pname in args.profiles:
            prof = PROFILES[pname]
            for sym in symbols:
                if adapter:
                    print(f"Updating {sym} {prof.interval} history ({prof.history_days} days)...", flush=True)
                    try:
                        update_history(adapter, sym, prof.interval, s.raw_dir, days=prof.history_days)
                    except Exception as exc:
                        print(f"  download failed ({type(exc).__name__}: {exc}); using what is on disk")
                df = load_candles(candle_path(s.raw_dir, "mexc_futures", sym, prof.interval))
                cutoff_old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=prof.history_days)
                df = df[df["timestamp"] >= cutoff_old].reset_index(drop=True)
                if len(df) < 500:
                    print(f"  {sym} {prof.interval}: only {len(df)} candles, skipping")
                    continue
                rep = validate_candles(df, prof.interval)
                days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
                data_notes.append(f"{prof.label}, {sym}: {len(df):,} candles over {days} days "
                                  f"({rep.missing_candles} missing)")
                print(f"  {sym} {prof.interval}: {len(df):,} candles, {days} days, testing...", flush=True)

                f = features(df, prof.interval, prof.trend_interval)
                ctx = regime_context(df, f)
                split_time = df["timestamp"].iloc[int(len(df) * DEV_SHARE)]
                for setup in SETUPS:
                    sig = signals(setup, df, f)
                    for tr in TARGETS_R:
                        cfg = RunConfig(target_r=tr, max_hold_bars=prof.max_hold_bars,
                                        limit_expiry_bars=prof.pullback_expiry)
                        trades, stats = run_backtest(df, sig, cfg, INTERVAL_SECONDS[prof.interval], ctx)
                        t = pd.DataFrame(trades)
                        key = dict(length=prof.label, length_id=pname, setup=setup, target_r=tr, symbol=sym)
                        if t.empty:
                            rows.append({**key, "eligible": False, "dev": summarize(t), "test": summarize(t),
                                         "all": summarize(t), "stats": stats, "regimes": {}})
                            continue
                        dev, test = t[t["entry_time"] < split_time], t[t["entry_time"] >= split_time]
                        ds = summarize(dev, cfg.start_equity)
                        ts_ = summarize(test, slice_start_equity(test, cfg.start_equity))
                        eligible = (ds["trades"] >= MIN_DEV_TRADES and ts_["trades"] >= MIN_TEST_TRADES
                                    and (ds["expectancy_r"] or 0) > 0 and (ts_["expectancy_r"] or 0) > 0
                                    and (ds["t_stat"] or 0) >= MIN_DEV_T)
                        rows.append({**key, "eligible": eligible, "dev": ds, "test": ts_,
                                     "all": summarize(t, cfg.start_equity), "stats": stats,
                                     "regimes": {**by_group(t, "trend_regime"), **by_group(t, "vol_regime")},
                                     "split": split_time})
                        t = t.assign(**key, period=["development" if x < split_time else "test"
                                                    for x in t["entry_time"]])
                        all_trades.append(t)
    finally:
        if adapter:
            adapter.close()

    if not rows:
        print("No results: no data available.")
        return
    write_reports(rows, all_trades, data_notes)


def tgt(r) -> str:
    return r.get("target_label") or f"{r['target_r']}R"


def write_reports(rows, all_trades, data_notes, title: str = "Strategy comparison",
                  prefix: str = "strategy", rules: str | None = None) -> None:
    rows.sort(key=lambda r: (r["dev"]["win_rate"] or -1), reverse=True)
    eligible = [r for r in rows if r["eligible"]]
    winner = eligible[0] if eligible else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    flat = []
    for r in rows:
        d, t, a = r["dev"], r["test"], r["all"]
        flat.append({
            "length": r["length_id"], "setup": r["setup"], "target": tgt(r), "symbol": r["symbol"],
            "eligible": r["eligible"],
            "dev_trades": d["trades"], "dev_win_rate": d["win_rate"], "dev_expectancy_r": d["expectancy_r"],
            "dev_profit_factor": d["profit_factor"], "dev_max_dd_pct": d["max_drawdown_pct"],
            "dev_t_stat": d["t_stat"],
            "test_trades": t["trades"], "test_win_rate": t["win_rate"], "test_expectancy_r": t["expectancy_r"],
            "test_profit_factor": t["profit_factor"], "test_max_dd_pct": t["max_drawdown_pct"],
            "all_return_pct": a["total_return_pct"], "costs_share_of_risk": a["costs_share_of_risk"],
            "limit_fill_rate": (r["stats"]["limit_filled"] / r["stats"]["limit_orders"]
                                if r["stats"].get("limit_orders") else None),
        })
    pd.DataFrame(flat).to_csv(out_dir / f"{prefix}_comparison_{stamp}.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(out_dir / f"{prefix}_trades_{stamp}.csv", index=False)

    lines = [f"# {title}, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    if rules:
        lines += [rules, ""]
    lines += ["All results are after fees (0% maker, 0.02% taker), 0.01% slippage on market and stop fills, "
              "and funding. Risk 1% of equity per trade. If a stop and a target fall in the same candle, "
              "the trade is counted as a loss.", "",
              f"Development = first {DEV_SHARE:.0%} of each history; Test = the last {1 - DEV_SHARE:.0%}, not used "
              "for ranking. Eligible = ≥30 development trades, ≥10 test trades, a positive average result "
              "per trade (expectancy) in both periods, and a development t-stat of at least 2 (the average "
              "result is at least two standard errors above zero, so it is unlikely to be luck).", ""]
    if winner:
        d, t = winner["dev"], winner["test"]
        lines += ["## Result", "",
                  f"**Highest win rate among eligible combinations: {winner['length']}, {winner['setup']}, "
                  f"target {tgt(winner)}, {winner['symbol']}.**", "",
                  f"Development: {d['trades']} trades, win rate {pct(d['win_rate'])}, "
                  f"expectancy {num(d['expectancy_r'])} R per trade, profit factor {num(d['profit_factor'])}, "
                  f"max drawdown {d['max_drawdown_pct']:.1f}%.  ",
                  f"Test: {t['trades']} trades, win rate {pct(t['win_rate'])}, "
                  f"expectancy {num(t['expectancy_r'])} R per trade, profit factor {num(t['profit_factor'])}, "
                  f"max drawdown {t['max_drawdown_pct']:.1f}%.", ""]
        if (t["win_rate"] or 0) < 0.6:
            lines += [f"This does not reach the 60% win-rate goal on the test period "
                      f"({pct(t['win_rate'])}).", ""]
    else:
        lines += ["## Result", "", "**No combination is eligible.** None made money after costs in both the "
                  "development and the test period with enough trades. The baseline rules do not have a "
                  "demonstrated edge yet; the next step is better features (Milestone 6) rather than trading these.", ""]

    lines += ["## All combinations (sorted by development win rate)", "",
              "| Trade length | Setup | Target | Symbol | Dev trades | Dev win | Dev exp. R | Dev t | Test trades | "
              "Test win | Test exp. R | Test PF | Eligible |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in rows:
        d, t = r["dev"], r["test"]
        lines.append(f"| {r['length']} | {r['setup']} | {tgt(r)} | {r['symbol']} | {d['trades']} | "
                     f"{pct(d['win_rate'])} | {num(d['expectancy_r'])} | {num(d['t_stat'], 1)} | {t['trades']} | "
                     f"{pct(t['win_rate'])} | "
                     f"{num(t['expectancy_r'])} | {num(t['profit_factor'])} | {'yes' if r['eligible'] else ''} |")

    lines += ["", "## Best per trade length (by test expectancy)", ""]
    for lid in dict.fromkeys(r["length_id"] for r in rows):
        group = [r for r in rows if r["length_id"] == lid and r["test"]["trades"] >= MIN_TEST_TRADES]
        if not group:
            continue
        b = max(group, key=lambda r: r["test"]["expectancy_r"] or -9)
        lines.append(f"- **{b['length']}**: {b['setup']}, {tgt(b)}, {b['symbol']}: test win rate "
                     f"{pct(b['test']['win_rate'])}, expectancy {num(b['test']['expectancy_r'])} R over "
                     f"{b['test']['trades']} trades.")

    if winner and winner["regimes"]:
        lines += ["", "## Winner by market regime (whole history)", "", "| Regime | Trades | Win rate | Exp. R |",
                  "|---|---:|---:|---:|"]
        for k, v in winner["regimes"].items():
            lines.append(f"| {k} | {v['trades']} | {pct(v['win_rate'])} | {num(v['expectancy_r'])} |")

    lines += ["", "## Data used", ""] + [f"- {n}" for n in data_notes]
    lines += ["", "Past results do not guarantee future results. Next step: paper trading the chosen "
              "combination live before any real money."]
    md = "\n".join(lines)
    path = out_dir / f"{prefix}_comparison_{stamp}.md"
    path.write_text(md, encoding="utf-8")

    # Console summary
    print("\n" + "=" * 100)
    sw = max(10, max(len(r["setup"]) for r in rows) + 2)
    print(f"{'Length':<10}{'Setup':<{sw}}{'Target':>8}  {'Symbol':<10}{'Dev n':>6}{'Dev win':>9}{'Dev expR':>10}"
          f"{'Dev t':>7}{'Test n':>8}{'Test win':>10}{'Test expR':>11}  Eligible")
    for r in rows:
        d, t = r["dev"], r["test"]
        print(f"{r['length_id']:<10}{r['setup']:<{sw}}{tgt(r):>8}  {r['symbol']:<10}{d['trades']:>6}"
              f"{pct(d['win_rate']):>9}{num(d['expectancy_r']):>10}{num(d['t_stat'], 1):>7}{t['trades']:>8}"
              f"{pct(t['win_rate']):>10}"
              f"{num(t['expectancy_r']):>11}  {'YES' if r['eligible'] else ''}")
    print("=" * (100 + sw - 10))
    if winner:
        print(f"WINNER: {winner['length']} | {winner['setup']} | {tgt(winner)} | {winner['symbol']}  "
              f"test win rate {pct(winner['test']['win_rate'])}, test expectancy "
              f"{num(winner['test']['expectancy_r'])} R")
    else:
        print("No combination made money after costs in both periods. See the report for details.")
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
