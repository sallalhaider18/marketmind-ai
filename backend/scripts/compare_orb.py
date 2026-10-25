"""Backtest the New York session opening-range breakout (ORB) in all its variants.

Run from the backend/ folder:
    python -m scripts.compare_orb                 # downloads 5m + 15m history (365 days) if needed
    python -m scripts.compare_orb --no-download

Variants: 5m / 15m candles x entry (close, retest) x stop (range, mid)
x target (1x, 2x, partial, 1:1, 1.5:1, 2:1) x BTC / ETH = 96 combinations.
Same costs, conservative fills and pass/fail rules as compare_strategies:
70% development / 30% test split in time; eligible = >=30 development trades,
>=10 test trades, profit after costs in both periods, development t-stat >= 2.
Rules are documented in app/strategy/orb.py.
"""
from __future__ import annotations

import argparse

import pandas as pd

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.backtest.metrics import by_group, summarize
from app.config import get_settings
from app.data.historical import candle_path, load_candles, update_history
from app.data.validation import validate_candles
from app.logging_setup import setup_logging
from app.strategy.orb import ENTRIES, STOPS, TARGETS, OrbConfig, run_orb
from scripts.compare_strategies import (DEV_SHARE, MIN_DEV_T, MIN_DEV_TRADES, MIN_TEST_TRADES,
                                        slice_start_equity, write_reports)

TIMEFRAMES = {"5m": 5, "15m": 15}
HISTORY_DAYS = 365

RULES = ("**Rules (New York time):** opening range = high/low of 9:30-9:45 ET on weekdays. Breakout = first "
         "candle closing outside the range that opens before 11:00 ET; one trade per day. Entry *close* = "
         "market at the next candle's open; entry *retest* = limit at the broken range edge until 11:00 ET "
         "(cancelled if the 1x target is reached first). Stop *range* = opposite side of the range; stop "
         "*mid* = middle of the range. Targets measured from the broken edge: *1x* / *2x* the range; "
         "*partial* = half at 1x, stop to entry, rest at 2x. Reward-to-risk targets from the entry: *1:1* = as far "
         "as the stop, *1.5:1* and *2:1* = 1.5x and 2x the stop distance. Everything still open closes at 16:00 ET.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES), choices=list(TIMEFRAMES))
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    symbols = args.symbols or s.symbols
    adapter = None if args.no_download else MexcFuturesAdapter(s.mexc_rest_url)

    rows, all_trades, notes = [], [], []
    try:
        for tf in args.timeframes:
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
                rep = validate_candles(df, tf)
                span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
                notes.append(f"{tf} candles, {sym}: {len(df):,} candles over {span} days "
                             f"({rep.missing_candles} missing)")
                print(f"  {sym} {tf}: {len(df):,} candles, {span} days, testing...", flush=True)
                split_time = df["timestamp"].iloc[int(len(df) * DEV_SHARE)]

                for entry in ENTRIES:
                    for stop in STOPS:
                        for target in TARGETS:
                            cfg = OrbConfig(entry=entry, stop=stop, target=target)
                            trades, stats = run_orb(df, TIMEFRAMES[tf], cfg)
                            t = pd.DataFrame(trades)
                            setup = f"{entry} entry, {stop} stop"
                            key = dict(length=f"NY ORB ({tf} candles)", length_id=f"orb {tf}", setup=setup,
                                       target_r=0.0, target_label=target, symbol=sym)
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
                            rows.append({**key, "eligible": eligible, "dev": ds, "test": ts_,
                                         "all": summarize(t, cfg.start_equity), "stats": stats,
                                         "regimes": {**by_group(t, "range_regime"), **by_group(t, "side"),
                                                     **by_group(t, "weekday")}})
                            all_trades.append(t.assign(**key, period=["development" if x < split_time else "test"
                                                                      for x in t["entry_time"]]))
    finally:
        if adapter:
            adapter.close()

    if not rows:
        print("No results: no data available.")
        return
    write_reports(rows, all_trades, notes, title="New York opening-range breakout comparison",
                  prefix="orb", rules=RULES)


if __name__ == "__main__":
    main()
