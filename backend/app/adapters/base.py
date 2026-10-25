"""Exchange-agnostic types. Every venue adapter converts its data into these."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# Standard interval names used everywhere inside MarketMind, in seconds.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "8h": 28800,
    "1d": 86400,
}

# Column layout for every candle DataFrame in the project.
CANDLE_COLUMNS = [
    "timestamp",      # candle open time, UTC (pandas Timestamp, tz-aware)
    "open", "high", "low", "close",
    "volume",         # base-asset volume (e.g. BTC), = contracts * contract_size
    "contracts",      # raw venue volume in contracts
    "quote_volume",   # quote-asset turnover (USDT)
]


@dataclass(frozen=True)
class ContractSpec:
    symbol: str            # internal symbol, e.g. BTCUSDT.P
    venue_symbol: str      # venue symbol, e.g. BTC_USDT
    contract_size: float   # base asset per contract
    tick_size: float       # minimum price increment
    min_volume: float      # minimum order size in contracts
    volume_step: float     # order size increment in contracts


@dataclass(frozen=True)
class Trade:
    symbol: str
    timestamp: datetime    # UTC
    price: float
    contracts: float
    side: str              # "buy" = aggressive buyer (taker bought), "sell" = aggressive seller


class MarketDataAdapter(ABC):
    """Interface each exchange adapter must implement."""

    venue: str

    @abstractmethod
    def to_venue_symbol(self, symbol: str) -> str: ...

    @abstractmethod
    def get_contract_spec(self, symbol: str) -> ContractSpec: ...

    @abstractmethod
    def fetch_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return candles with open time in [start, end), columns = CANDLE_COLUMNS."""
