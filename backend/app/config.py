"""Application settings, loaded from environment variables or a .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# marketmind-ai/  (two levels above backend/app/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="MM_", extra="ignore"
    )

    # Instruments, written the way TradingView shows them (e.g. BTCUSDT.P)
    symbols: list[str] = ["BTCUSDT.P", "ETHUSDT.P"]
    default_interval: str = "1m"

    # MEXC futures public endpoints (no API key needed)
    mexc_rest_url: str = "https://contract.mexc.com"
    mexc_ws_url: str = "wss://contract.mexc.com/edge"

    # Live stream behaviour
    live_stream_enabled: bool = True
    ws_ping_seconds: float = 15.0
    stale_after_seconds: float = 10.0
    reconnect_max_seconds: float = 60.0

    # Background REST sync of 1m history for the dashboard
    history_sync_enabled: bool = True
    history_sync_seconds: float = 300.0
    history_days: int = 30

    # Order book (Milestone 3)
    orderbook_enabled: bool = True
    orderbook_snapshot_limit: int = 1000
    orderbook_refresh_seconds: float = 120.0   # rebuild from a fresh snapshot this often
    orderbook_record: bool = True          # save 1 snapshot/second to data/orderbook
    orderbook_record_levels: int = 25

    # Order-book trade plans (4h levels + strongest wall, checked every 30 min in sessions)
    plan_enabled: bool = True
    signals_enabled: bool = True          # order-flow signal engine (strategies A and B, paper only)

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"

    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
