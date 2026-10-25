"""Print the live MEXC perpetual feed in the terminal (no web server needed).

Run from the backend/ folder:
    python -m scripts.stream_live
Stop with Ctrl+C.
"""
from __future__ import annotations

import asyncio
import contextlib

from app.config import get_settings
from app.data.live_stream import LiveMarketState, MexcFuturesStream
from app.logging_setup import setup_logging


async def main() -> None:
    s = get_settings()
    setup_logging(s.log_level)
    state = LiveMarketState(s.symbols, stale_after=s.stale_after_seconds)
    stream = MexcFuturesStream(state, url=s.mexc_ws_url, interval=s.default_interval, ping_seconds=s.ws_ping_seconds)
    task = asyncio.create_task(stream.run())
    try:
        while True:
            await asyncio.sleep(2)
            for sym in state.symbols:
                snap = state.snapshot(sym)
                flow = snap["trade_flow_60s"]
                ratio = f"{flow['buy_ratio']:.0%}" if flow["buy_ratio"] is not None else "-"
                flag = "STALE" if snap["stale"] else "live "
                print(
                    f"{flag} {sym:<10} price {snap['last_price']}  "
                    f"60s trades {flow['trades']:>4}  buy share {ratio:>4}  "
                    f"delta {flow['delta_contracts']:+.0f} contracts"
                )
            print("-" * 80)
    finally:
        stream.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
