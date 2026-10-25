import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config
from app.data.historical import candle_path
from tests.test_validation_and_history import make_candles


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_LIVE_STREAM_ENABLED", "false")
    monkeypatch.setenv("MM_HISTORY_SYNC_ENABLED", "false")
    monkeypatch.setenv("MM_DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    from app.main import app
    with TestClient(app) as c:
        yield c, tmp_path
    config.get_settings.cache_clear()


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["real_order_execution"] == "disabled"


def test_symbols_and_unknown_symbol(client):
    c, _ = client
    assert "BTCUSDT.P" in c.get("/api/market/symbols").json()["symbols"]
    assert c.get("/api/market/candles/DOGEUSDT.P").status_code == 404


def test_candles_and_quality(client):
    c, tmp = client
    p = candle_path(tmp / "raw", "mexc_futures", "BTCUSDT.P", "1m")
    p.parent.mkdir(parents=True)
    make_candles(20).to_parquet(p, index=False)

    r = c.get("/api/market/candles/BTCUSDT.P?limit=5")
    assert r.status_code == 200 and len(r.json()["candles"]) == 5
    q = c.get("/api/market/quality/BTCUSDT.P").json()
    assert q["ok"] and q["rows"] == 20


def test_live_disabled(client):
    c, _ = client
    assert c.get("/api/market/live/BTCUSDT.P").status_code == 503
    assert c.get("/api/market/status").json() == {"live_stream": "disabled"}
