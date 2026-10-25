"""Check the MEXC order-book feed for 30 seconds and print a health report.

Run from the backend/ folder:
    python -m scripts.check_orderbook
    python -m scripts.check_orderbook --symbol ETHUSDT.P --seconds 60
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from collections import Counter

import httpx
import websockets

from app.adapters.mexc_futures import to_venue_symbol
from app.config import get_settings
from app.orderbook.book import BookSync


async def main(symbol: str, seconds: int) -> None:
    s = get_settings()
    vs = to_venue_symbol(symbol)
    sync = BookSync(symbol)
    gaps = Counter()
    last_v = None
    pushes = 0
    first_raw = None

    async with httpx.AsyncClient(base_url=s.mexc_rest_url, timeout=30) as client:
        spec = None
        for attempt in range(1, 4):
            try:
                spec = (await client.get("/api/v1/contract/detail", params={"symbol": vs})).json()["data"]
                break
            except httpx.HTTPError as exc:
                print(f"Request attempt {attempt}/3 to {s.mexc_rest_url} failed: {type(exc).__name__}")
                if attempt < 3:
                    await asyncio.sleep(5)
        if spec is None:
            print("\nRESULT: Python could not reach MEXC's API. If your browser can open "
                  "https://contract.mexc.com/api/v1/contract/ping, something on this computer (VPN, proxy, "
                  "antivirus or firewall) treats Python differently from the browser.")
            return
        cs = float(spec["contractSize"])
        snap_task = None
        ws_conn = None
        for attempt in range(1, 4):
            try:
                ws_conn = await websockets.connect(s.mexc_ws_url, ping_interval=None, open_timeout=30)
                break
            except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
                print(f"Connection attempt {attempt}/3 to {s.mexc_ws_url} failed: {type(exc).__name__}: {exc}")
                if attempt < 3:
                    await asyncio.sleep(5)
        if ws_conn is None:
            print("\nRESULT: could not connect to MEXC's live feed. Check your internet connection, VPN or "
                  "firewall, then try again. Paste this output to Claude if it keeps happening.")
            return
        async with ws_conn as ws:
            await ws.send(json.dumps({"method": "sub.depth", "param": {"symbol": vs}}))
            end = time.monotonic() + seconds
            next_ping = time.monotonic() + 15
            print(f"Listening to {vs} depth for {seconds} s ...")
            while time.monotonic() < end:
                if time.monotonic() > next_ping:
                    await ws.send(json.dumps({"method": "ping"}))
                    next_ping += 15
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if msg.get("channel") != "push.depth":
                    continue
                pushes += 1
                first_raw = first_raw or raw
                v = int(msg["data"]["version"])
                if last_v is not None:
                    gaps[v - last_v] += 1
                last_v = v
                sync.on_update(msg["data"])
                if sync.needs_snapshot and snap_task is None:
                    sync.snapshot_pending = True

                    async def fetch():
                        await asyncio.sleep(0.5)
                        r = (await client.get(f"/api/v1/contract/depth/{vs}",
                                              params={"limit": s.orderbook_snapshot_limit})).json()
                        d = r["data"]
                        ok = sync.on_snapshot(d["bids"], d["asks"], int(d["version"]))
                        print(f"Snapshot v{d['version']}: {len(d['bids'])} bids, {len(d['asks'])} asks, "
                              f"synced={ok}")
                        if not ok:
                            print("  reason:", sync.last_resync_reason)
                    snap_task = asyncio.create_task(fetch())
                elif snap_task and snap_task.done() and not sync.synced and sync.needs_snapshot:
                    snap_task = None  # try again

            # Final self-check: rebuild from a fresh snapshot and compare with the live book.
            drift = None
            if sync.synced:
                for _ in range(3):
                    r = (await client.get(f"/api/v1/contract/depth/{vs}",
                                          params={"limit": s.orderbook_snapshot_limit})).json()["data"]
                    # keep reading updates briefly so the live book catches up with the snapshot
                    t_end = time.monotonic() + 1.5
                    while time.monotonic() < t_end:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
                        except asyncio.TimeoutError:
                            break
                        if msg.get("channel") == "push.depth":
                            sync.on_update(msg["data"])
                    drift = sync.refresh(r["bids"], r["asks"], int(r["version"]))
                    if drift is not None:
                        break

    print("\nFirst depth message (trimmed):", first_raw[:300] if first_raw else None)
    print(f"Depth messages received: {pushes}")
    print("Version step between messages (step: count):", dict(sorted(gaps.items())[:12]))
    b = sync.book
    print(f"Synced at end: {sync.synced}   resyncs: {sync.resyncs}   last reason: {sync.last_resync_reason}")
    if sync.synced:
        print(f"Best bid {b.best_bid}  best ask {b.best_ask}  spread {b.best_ask - b.best_bid:.4g}")
        print(f"Levels: {len(b.bids)} bids, {len(b.asks)} asks; snapshot covers ±{b.coverage_pct() * 100:.2f}% of price")
        bids, asks = b.top(5)
        print("Top asks:", [(p, round(q * cs, 4)) for p, q in reversed(asks)])
        print("Top bids:", [(p, round(q * cs, 4)) for p, q in bids])
    print("Live book vs fresh snapshot (top 20 levels each side):",
          "could not compare" if drift is None else f"{drift:.0%} of levels differ")
    steps_ok = all(step > 0 for step in gaps)
    verdict = sync.synced and sync.resyncs == 0 and steps_ok and drift is not None and drift <= 0.05
    print("\nRESULT:", "OK, the order book stays in sync" if verdict else "NEEDS ATTENTION, paste this output to Claude")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT.P")
    p.add_argument("--seconds", type=int, default=30)
    a = p.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(a.symbol, a.seconds))
