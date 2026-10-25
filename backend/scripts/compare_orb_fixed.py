"""New York opening-range breakout with a FIXED stop and target.

Default: stop BTC 500 / ETH 50 (or 0.6%), target 2x the stop (BTC 1,000 / ETH 100).
Change with --btc, --eth, --pct and --rr (e.g. --rr 1 for target = stop).

Run from the backend/ folder:
    python -m scripts.compare_orb_fixed
    python -m scripts.compare_orb_fixed --no-download
    python -m scripts.compare_orb_fixed --btc 1000 --eth 100 --pct 1.2

The breakout decides when and which way to enter (see app/strategy/orb.py);
then the stop and the target are the same distance from the entry:
  * "your sizes":  BTC +/- 1,000, ETH +/- 100 (changeable with --btc / --eth)
  * "equal %":     both +/- 1.2% of the entry (about 1,000 on BTC and 30 on ETH at today's prices)
Every day is traded, Saturday and Sunday included. A trade ends at the target,
at the stop, or, if neither is hit, at 9:29 ET the next day (one minute before
the next New York session). So there is never more than one trade open.
Variants: 5m / 15m candles x entry (close, retest) x size x BTC / ETH = 16.
Same costs, conservative fills and pass/fail rules as the other comparisons.
"""
from __future__ import annotations

import argparse

import pandas as pd

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.backtest.metrics import by_group, summarize
from app.config import get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.logging_setup import setup_logging
from app.strategy.orb import ENTRIES, OrbConfig, run_orb
from scripts.compare_strategies import (DEV_SHARE, MIN_DEV_T, MIN_DEV_TRADES, MIN_TEST_TRADES,
                                        slice_start_equity, write_reports)

TIMEFRAMES = {"5m": 5, "15m": 15}
HISTORY_DAYS = 365


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--btc", type=float, default=500.0, help="BTC stop distance in USDT")
    p.add_argument("--eth", type=float, default=50.0, help="ETH stop distance in USDT")
    p.add_argument("--rr", type=float, default=2.0, help="target = rr x stop (2 = stop 500, target 1000)")
    p.add_argument("--pct", type=float, default=0.6, help="equal-%% variant: stop as percent of entry")
    p.add_argument("--weekdays-only", action="store_true", help="skip Saturday and Sunday")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    symbols = args.symbols or s.symbols
    adapter = None if args.no_download else MexcFuturesAdapter(s.mexc_rest_url)
    distance = {sym: (args.btc if sym.upper().startswith("BTC") else args.eth) for sym in symbols}

    rows, all_trades, notes = [], [], []
    try:
        for tf, minutes in TIMEFRAMES.items():
            for sym in symbols:
                if adapter:
                    print(f"Updating {sym} {tf} history ({HISTORY_DAYS} days)...", flush=True)
                    try:
                        update_history(adapter, sym, tf, s.raw_dir, days=HISTORY_DAYS)
                    except Exception as exc:
                        print(f"  download failed ({type(exc).__name__}: {exc}); using what is on disk")
                df = load_candles(candle_path(s.raw_dir, "mexc_futures", sym, tf))
                df = df[df["timestamp"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HISTORY_DAYS)]
                df = df.reset_index(drop=True)
                if len(df) < 1000:
                    print(f"  {sym} {tf}: only {len(df)} candles, skipping")
                    continue
                span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
                notes.append(f"{tf} candles, {sym}: {len(df):,} over {span} days, price "
                             f"{df['close'].min():,.0f}-{df['close'].max():,.0f}")
                print(f"  {sym} {tf}: {len(df):,} candles, {span} days, testing...", flush=True)
                split_time = df["timestamp"].iloc[int(len(df) * DEV_SHARE)]

                d = distance[sym]
                sizes = {f"SL {d:,.0f} TP {d * args.rr:,.0f}": {"fixed_distance": d, "fixed_rr": args.rr},
                         f"SL {args.pct:g}% TP {args.pct * args.rr:g}%": {"fixed_pct": args.pct / 100,
                                                                         "fixed_rr": args.rr}}
                exits = {"until next day 9:29": {"exit_next_open": True, "weekends": not args.weekdays_only}}
                for entry in ENTRIES:
                    for size_name, size_kw in sizes.items():
                        for exit_name, exit_kw in exits.items():
                            cfg = OrbConfig(entry=entry, stop="range", target="1x", **exit_kw, **size_kw)
                            trades, stats = run_orb(df, minutes, cfg)
                            t = pd.DataFrame(trades)
                            key = dict(length=f"NY ORB fixed stop/target ({tf})", length_id=f"orb {tf}",
                                       setup=f"{entry} entry, {exit_name}", target_r=args.rr,
                                       target_label=size_name, symbol=sym)
                            stats = {**stats, "limit_orders": 0}
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
                            exits_mix = t["exit_reason"].value_counts(normalize=True).to_dict()
                            rows.append({**key, "eligible": eligible, "dev": ds, "test": ts_,
                                         "all": summarize(t, cfg.start_equity), "stats": stats,
                                         "regimes": {**by_group(t, "exit_reason"), **by_group(t, "side")},
                                         "exit_mix": exits_mix})
                            all_trades.append(t.assign(**key, period=["development" if x < split_time else "test"
                                                                      for x in t["entry_time"]]))
    finally:
        if adapter:
            adapter.close()

    if not rows:
        print("No results: no data available.")
        return
    be = 1 / (1 + args.rr) * 100
    rules = (f"**Fixed stop and target (target {args.rr:g}x the stop):** the New York opening-range breakout decides when and which way to enter "
             "(first candle closing outside the 9:30-9:45 ET range, before 11:00 ET; entry at the next open, or "
             "on a retest of the range edge). Then the stop is BTC "
             f"{args.btc:,.0f} / ETH {args.eth:,.0f} (or {args.pct:g}%) from the entry and the target {args.rr:g}x as "
             f"far. Break-even win rate before costs: {be:.0f}%. A trade ends at the target, the stop, or at "
             "9:29 ET the next day if neither is hit. "
             + ("Weekdays only." if args.weekdays_only else "Every day is traded, Saturday and Sunday included."))
    write_reports(rows, all_trades, notes, title="New York breakout with a fixed stop and target",
                  prefix="orb_fixed", rules=rules)
    print("\nHow trades ended (share of all trades):")
    for r in rows:
        mix = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r.get("exit_mix", {}).items(), key=lambda x: -x[1]))
        print(f"  {r['length_id']:<8} {r['setup']:<34} {r['target_label']:<20} {r['symbol']:<10} {mix}")


if __name__ == "__main__":
    main()
