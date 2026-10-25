"""Download / update historical candles for the configured MEXC perpetuals and
print a data-quality report for each.

Run from the backend/ folder:
    python -m scripts.download_history                     # 30 days of 1m for BTC + ETH
    python -m scripts.download_history --days 365 --interval 15m
    python -m scripts.download_history --symbols BTCUSDT.P
"""
from __future__ import annotations

import argparse
import json

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.config import get_settings
from app.data.historical import update_history
from app.data.validation import validate_candles
from app.logging_setup import setup_logging


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=s.symbols)
    p.add_argument("--interval", default=s.default_interval)
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()
    setup_logging(s.log_level)

    adapter = MexcFuturesAdapter(s.mexc_rest_url)
    try:
        for sym in args.symbols:
            spec = adapter.get_contract_spec(sym)
            print(f"\n=== {sym} ({spec.venue_symbol}) contract size {spec.contract_size}, tick {spec.tick_size} ===")
            path, df = update_history(adapter, sym, args.interval, s.raw_dir, days=args.days)
            report = validate_candles(df, args.interval)
            print(f"Saved {len(df)} candles -> {path}")
            print(report.summary())
            report_path = path.with_name(f"{args.interval}.quality.json")
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
