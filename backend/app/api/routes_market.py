from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.adapters.base import INTERVAL_SECONDS
from app.config import get_settings
from app.data.historical import candle_path, load_candles
from app.data.validation import validate_candles

router = APIRouter(prefix="/api/market", tags=["market"])
VENUE = "mexc_futures"


def _check_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s not in [x.upper() for x in get_settings().symbols]:
        raise HTTPException(404, f"Unknown symbol {symbol}. Configured: {get_settings().symbols}")
    return s


@router.get("/symbols")
def symbols():
    return {"venue": VENUE, "symbols": get_settings().symbols}


@router.get("/candles/{symbol}")
def candles(symbol: str, interval: str = "1m", limit: int = Query(500, ge=1, le=10000)):
    """Most recent stored historical candles (run scripts/download_history.py first)."""
    s = _check_symbol(symbol)
    if interval not in INTERVAL_SECONDS:
        raise HTTPException(400, f"interval must be one of {list(INTERVAL_SECONDS)}")
    df = load_candles(candle_path(get_settings().raw_dir, VENUE, s, interval))
    if df.empty:
        raise HTTPException(404, "No stored candles. Run: python -m scripts.download_history")
    df = df.tail(limit).copy()
    df["timestamp"] = df["timestamp"].map(lambda t: t.isoformat())
    return {"symbol": s, "interval": interval, "candles": df.to_dict(orient="records")}


@router.get("/quality/{symbol}")
def quality(symbol: str, interval: str = "1m"):
    """Data-quality report for the stored historical candles."""
    s = _check_symbol(symbol)
    df = load_candles(candle_path(get_settings().raw_dir, VENUE, s, interval))
    return validate_candles(df, interval).to_dict()


@router.get("/live/{symbol}")
def live(symbol: str, request: Request):
    """Latest live candle, last price, 60 s aggressive buy/sell flow, and feed health."""
    s = _check_symbol(symbol)
    state = getattr(request.app.state, "live", None)
    if state is None:
        raise HTTPException(503, "Live stream disabled (MM_LIVE_STREAM_ENABLED=false)")
    return state.snapshot(s)


@router.get("/status")
def status(request: Request):
    state = getattr(request.app.state, "live", None)
    if state is None:
        return {"live_stream": "disabled"}
    return {
        "live_stream": "connected" if state.connected else "disconnected",
        "reconnects": state.reconnects,
        "last_error": state.last_error,
        "symbols": {s: {"stale": state.is_stale(s)} for s in state.symbols},
    }
