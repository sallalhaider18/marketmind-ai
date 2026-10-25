"""MarketMind AI backend.

Run from the backend/ folder:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000        (dashboard, after `npm run build` in frontend/)
     or http://127.0.0.1:8000/docs   (API explorer)
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.adapters.mexc_futures import MexcFuturesAdapter
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_market import router as market_router
from app.config import PROJECT_ROOT, get_settings
from app.data.candle_store import CandleStore
from app.data.live_stream import LiveMarketState, MexcFuturesStream
from app.logging_setup import setup_logging
from app.obtrade.engine import PlanEngine
from app.orderbook.engine import OrderBookEngine
from app.orderbook.recorder import OrderBookRecorder
from app.signals.rows import RowRecorder
from app.signals.runner import SignalRunner
from app.sranalyzer.analyzer import LevelStore, SRAnalyzer

log = logging.getLogger("marketmind")
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    tasks: list[asyncio.Task] = []
    stream = adapter = None
    app.state.live = None
    app.state.store = None
    app.state.orderbook = None
    app.state.plan = None
    app.state.signals = None
    app.state.sr = None

    if settings.live_stream_enabled:
        state = LiveMarketState(settings.symbols, stale_after=settings.stale_after_seconds)
        stream = MexcFuturesStream(
            state,
            url=settings.mexc_ws_url,
            interval="1m",
            ping_seconds=settings.ws_ping_seconds,
            reconnect_max_seconds=settings.reconnect_max_seconds,
            rest_url=settings.mexc_rest_url,
            depth_enabled=settings.orderbook_enabled,
            snapshot_limit=settings.orderbook_snapshot_limit,
            refresh_seconds=settings.orderbook_refresh_seconds,
        )
        app.state.live = state
        tasks.append(asyncio.create_task(stream.run()))
        log.info("Live stream started for %s", settings.symbols)

    if settings.history_sync_enabled:
        adapter = MexcFuturesAdapter(settings.mexc_rest_url)
        store = CandleStore(adapter, settings.symbols, settings.raw_dir, live=app.state.live,
                            days=settings.history_days)
        app.state.store = store
        tasks.append(asyncio.create_task(store.run_sync_loop(settings.history_sync_seconds)))

    if settings.orderbook_enabled and app.state.live is not None:
        recorder = None
        if settings.orderbook_record:
            recorder = OrderBookRecorder(settings.data_dir / "orderbook" / "mexc_futures",
                                         levels=settings.orderbook_record_levels)
        store = app.state.store
        signals = None
        if settings.signals_enabled:
            if adapter is None:
                adapter = MexcFuturesAdapter(settings.mexc_rest_url)
            from app.data.historical import update_history
            sig_adapter, sig_raw = adapter, settings.raw_dir
            signals = SignalRunner(
                settings.symbols,
                candles=lambda s, iv: update_history(sig_adapter, s, iv, sig_raw,
                                                     days={"4h": 120, "1d": 365}[iv])[1],
                reports=PROJECT_ROOT / "reports",
                recorder=RowRecorder(settings.data_dir / "signals" / "mexc_futures") if settings.orderbook_record else None,
            )
        sig_ = signals
        sr = SRAnalyzer(
            settings.symbols,
            LevelStore(PROJECT_ROOT / "reports" / "my_levels.json"),
            reports=PROJECT_ROOT / "reports",
            raw_dir=(settings.data_dir / "sr_analyzer" / "mexc_futures") if settings.orderbook_record else None,
            auto_levels=(lambda s: sig_.levels.get(s, [])) if sig_ else None,
        )
        app.state.sr = sr
        engine = OrderBookEngine(
            app.state.live,
            contract_size=lambda s: store.info(s)["contract_size"] if store else None,
            recorder=recorder,
            on_row=signals.on_row if signals else None,
            on_tick=sr.on_tick,
        )
        app.state.orderbook = engine
        app.state.signals = signals
        tasks.append(asyncio.create_task(engine.run()))
        if signals:
            live_ = app.state.live
            tasks.append(asyncio.create_task(signals.run_levels(
                lambda s: live_.symbols[s].book.book.mid if live_.symbols[s].book.synced else None)))
            tasks.append(asyncio.create_task(signals.run_recorder()))
            log.info("Order-flow signal engine running (strategies A and B, paper only)")

    if settings.plan_enabled and app.state.live is not None:
        if adapter is None:
            adapter = MexcFuturesAdapter(settings.mexc_rest_url)
        from app.data.historical import update_history
        plan_adapter, raw_dir, store_ = adapter, settings.raw_dir, app.state.store
        plan = PlanEngine(
            app.state.live,
            contract_size=lambda s: store_.info(s)["contract_size"] if store_ else None,
            candles=lambda s, iv: update_history(plan_adapter, s, iv, raw_dir,
                                                 days={"4h": 120, "1d": 365}[iv])[1],
            state_path=PROJECT_ROOT / "reports" / "ob_plan_state.json",
            log_path=PROJECT_ROOT / "reports" / "ob_trades.csv",
        )
        app.state.plan = plan
        tasks.append(asyncio.create_task(plan.run()))
        log.info("Order-book trade plans: checks every 30 min during Asia, London and New York sessions")

    yield

    if stream:
        stream.stop()
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t
    if adapter:
        adapter.close()


app = FastAPI(title="MarketMind AI", version="0.2.0", lifespan=lifespan)
app.include_router(market_router)
app.include_router(dashboard_router)


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "real_order_execution": "disabled"}


class DashboardFiles(StaticFiles):
    """Built dashboard. The page itself is never cached, so after an update the
    browser always loads the new version; hashed asset files can be cached."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


# Serve the built dashboard (frontend/dist) at "/" when it exists.
if FRONTEND_DIST.is_dir():
    app.mount("/", DashboardFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
