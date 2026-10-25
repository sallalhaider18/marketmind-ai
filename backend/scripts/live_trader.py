"""Live PAPER trading in the terminal with the trained ML / deep-learning models.
No real orders are ever placed.

Run from the backend/ folder (after training with: python -m scripts.direction_ml):
    python -m scripts.live_trader
    python -m scripts.live_trader --model ml        # or dl, ensemble (default: chosen by training)
    python -m scripts.live_trader --reset           # start a fresh paper account

Every 30-minute candle close in the New York session (10:00 to 16:00 NY time,
every day) it prints each coin's model reading and whether it trades. Entries,
exits and results are printed and saved to reports/paper_trades.csv. Leave it
running; stop it with Ctrl+C (open trades are remembered for next time).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.config import PROJECT_ROOT, get_settings
from app.direction.live import MexcMarket, PaperTrader, Predictor
from app.logging_setup import setup_logging

MODEL_DIR = PROJECT_ROOT / "models" / "direction"
STATE = PROJECT_ROOT / "reports" / "paper_state.json"
LOG = PROJECT_ROOT / "reports" / "paper_trades.csv"
LABELS = {"ml": "ML (gradient-boosted trees)", "dl": "Deep learning (LSTM)", "ensemble": "ML + DL combined"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ml", "dl", "ensemble"], default=None)
    ap.add_argument("--reset", action="store_true", help="delete the paper account and start again")
    ap.add_argument("--every", type=float, default=15.0, help="seconds between checks")
    args = ap.parse_args()
    s = get_settings()
    setup_logging("WARNING")

    try:
        pred = Predictor(MODEL_DIR, args.model)
    except FileNotFoundError as exc:
        print(exc)
        return
    if args.reset and STATE.exists():
        STATE.unlink()
        print("Paper account reset.")

    adapter = MexcFuturesAdapter(s.mexc_rest_url)
    market = MexcMarket(adapter, list(pred.distance), s.raw_dir)
    trader = PaperTrader(market, pred, pred.model, pred.threshold, pred.distance, STATE, LOG,
                         out=lambda line: print(line, flush=True))

    print("=" * 100)
    print("MarketMind live PAPER trading (no real orders)")
    print(f"Model: {LABELS[pred.model]}   trades when up chance >= {0.5 + pred.threshold:.0%} (long) "
          f"or <= {0.5 - pred.threshold:.0%} (short)")
    print("Stop = target: " + ", ".join(f"{k} ±{v:,.0f}" for k, v in pred.distance.items())
          + "   one trade per coin per day, closed by 9:29 NY next day at the latest")
    print(f"Models trained up to {pred.meta['trained_until'][:16]} UTC")
    if not pred.proven:
        print("NOTE: this model did NOT pass the backtest checks. Treat these paper trades as an experiment.")
    print(f"Trades are saved to {LOG}")
    print("=" * 100, flush=True)

    last_status = 0.0
    try:
        while True:
            now = datetime.now(timezone.utc)
            try:
                trader.step(now)
            except Exception as exc:   # network hiccups etc.: report and keep going
                print(f"(temporary problem: {type(exc).__name__}: {exc}; retrying)", flush=True)
            if time.time() - last_status > 600:
                print(trader.status(now), flush=True)
                last_status = time.time()
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nStopped. " + trader.status(datetime.now(timezone.utc)))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
