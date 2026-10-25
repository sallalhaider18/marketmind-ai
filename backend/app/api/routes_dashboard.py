"""Endpoints used by the dashboard: chart history and a live WebSocket."""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.data.candle_store import to_chart_bars

log = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])

CHART_INTERVALS = ["30m", "4h", "1d"]


def _symbol(symbol: str) -> str:
    s = symbol.upper()
    if s not in [x.upper() for x in get_settings().symbols]:
        raise HTTPException(404, f"Unknown symbol {symbol}")
    return s


def _interval(interval: str) -> str:
    if interval not in CHART_INTERVALS:
        raise HTTPException(400, f"interval must be one of {CHART_INTERVALS}")
    return interval


def _feed(app, symbol: str) -> dict:
    live = app.state.live
    store = app.state.store
    feed = live.snapshot(symbol) if live else {"connected": False, "stale": True, "live_stream": "disabled"}
    if live:
        feed["reconnects"] = live.reconnects
        feed["last_error"] = live.last_error
    feed["history"] = store.info(symbol) if store else None
    return feed


@router.get("/api/dashboard/config")
def dashboard_config():
    return {"symbols": get_settings().symbols, "intervals": CHART_INTERVALS}


@router.get("/api/dashboard/chart/{symbol}")
def chart(request: Request, symbol: str, interval: str = "4h", limit: int = Query(1000, ge=10, le=5000)):
    s, iv = _symbol(symbol), _interval(interval)
    store = request.app.state.store
    if store is None:
        raise HTTPException(503, "Candle store not running")
    bars = to_chart_bars(store.candles(s, iv, limit))
    return {"symbol": s, "interval": iv, "bars": bars, "feed": _feed(request.app, s)}


class LevelIn(BaseModel):
    price: float = Field(gt=0)


def _sr(request: Request):
    sr = getattr(request.app.state, "sr", None)
    if sr is None:
        raise HTTPException(503, "S/R analyzer not running (needs the live order book)")
    return sr


@router.get("/api/levels/{symbol}")
def my_levels(symbol: str, request: Request) -> dict:
    s = _symbol(symbol)
    return {"symbol": s, "levels": _sr(request).store.get(s)}


@router.post("/api/levels/{symbol}")
def add_level(symbol: str, body: LevelIn, request: Request) -> dict:
    s = _symbol(symbol)
    return {"symbol": s, "levels": _sr(request).store.add(s, body.price)}


@router.delete("/api/levels/{symbol}")
def remove_level(symbol: str, request: Request, price: float = Query(gt=0)) -> dict:
    s = _symbol(symbol)
    return {"symbol": s, "levels": _sr(request).store.remove(s, price)}


@router.get("/api/dashboard/orderbook/{symbol}")
def orderbook(request: Request, symbol: str):
    s = _symbol(symbol)
    engine = getattr(request.app.state, "orderbook", None)
    if engine is None:
        raise HTTPException(503, "Order book disabled")
    return engine.views.get(s)


@router.websocket("/ws/live/{symbol}")
async def live_ws(ws: WebSocket, symbol: str, interval: str = "4h"):
    """Pushes once per second: the last two bars (so a bar rolling over is
    captured) plus feed health and 60 s trade flow."""
    s = symbol.upper()
    if s not in [x.upper() for x in get_settings().symbols] or interval not in CHART_INTERVALS:
        await ws.close(code=1008)
        return
    await ws.accept()
    app = ws.app
    try:
        while True:
            store = app.state.store
            bars = to_chart_bars(store.candles(s, interval, 2)) if store else []
            ob = app.state.orderbook.views.get(s) if getattr(app.state, "orderbook", None) else None
            plan = app.state.plan.view(s) if getattr(app.state, "plan", None) else None
            signals = app.state.signals.view(s) if getattr(app.state, "signals", None) else None
            sr = app.state.sr.view(s) if getattr(app.state, "sr", None) else None
            await ws.send_json({"symbol": s, "interval": interval, "bars": bars, "feed": _feed(app, s),
                                "orderbook": ob, "plan": plan, "signals": signals, "sr": sr})
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        pass
