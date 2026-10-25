"""MEXC USDT-margined perpetual futures adapter (public market data only).

Docs: https://mexcdevelop.github.io/apidocs/contract_v1_en/
  REST base:  https://contract.mexc.com
  K-lines:    GET /api/v1/contract/kline/{symbol}?interval=Min1&start=<sec>&end=<sec>
              max 2000 candles per request, 20 requests / 2 s
  Contracts:  GET /api/v1/contract/detail?symbol=BTC_USDT   (1 request / 5 s)

No API key is used or needed. This adapter cannot place orders.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from .base import CANDLE_COLUMNS, INTERVAL_SECONDS, ContractSpec, MarketDataAdapter

log = logging.getLogger(__name__)

# MarketMind interval -> MEXC futures interval name
MEXC_INTERVALS = {
    "1m": "Min1",
    "5m": "Min5",
    "15m": "Min15",
    "30m": "Min30",
    "1h": "Min60",
    "4h": "Hour4",
    "8h": "Hour8",
    "1d": "Day1",
}

MAX_CANDLES_PER_REQUEST = 2000
MIN_SECONDS_BETWEEN_REQUESTS = 0.15  # stays under 20 req / 2 s


class MexcApiError(RuntimeError):
    pass


def to_venue_symbol(symbol: str) -> str:
    """'BTCUSDT.P' / 'BTCUSDT' / 'BTC_USDT' -> 'BTC_USDT'."""
    s = symbol.upper().strip()
    if s.endswith(".P"):
        s = s[:-2]
    if "_" in s:
        return s
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[: -len(quote)]}_{quote}"
    raise ValueError(f"Cannot map symbol {symbol!r} to a MEXC futures symbol")


def parse_kline_response(payload: dict, contract_size: float) -> pd.DataFrame:
    """Convert MEXC's column-array kline response into a standard candle DataFrame."""
    if not payload.get("success", False):
        raise MexcApiError(f"MEXC error {payload.get('code')}: {payload.get('message')}")
    data = payload.get("data") or {}
    times = data.get("time") or []
    if not times:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    contracts = pd.Series(data["vol"], dtype="float64")
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series(times, dtype="int64"), unit="s", utc=True),
            "open": pd.Series(data["open"], dtype="float64"),
            "high": pd.Series(data["high"], dtype="float64"),
            "low": pd.Series(data["low"], dtype="float64"),
            "close": pd.Series(data["close"], dtype="float64"),
            "volume": contracts * contract_size,
            "contracts": contracts,
            "quote_volume": pd.Series(data.get("amount", [float("nan")] * len(times)), dtype="float64"),
        }
    )
    return df[CANDLE_COLUMNS]


class MexcFuturesAdapter(MarketDataAdapter):
    venue = "mexc_futures"

    def __init__(self, base_url: str = "https://contract.mexc.com", client: httpx.Client | None = None):
        self._client = client or httpx.Client(base_url=base_url, timeout=15.0)
        self._specs: dict[str, ContractSpec] = {}
        self._last_request = 0.0

    # -- helpers -----------------------------------------------------------
    def to_venue_symbol(self, symbol: str) -> str:
        return to_venue_symbol(symbol)

    def _get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(retries):
            self._last_request = time.monotonic()
            try:
                r = self._client.get(path, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt == retries - 1:
                    raise
                backoff = 2 ** attempt
                log.warning("MEXC request %s failed (%s); retrying in %ss", path, exc, backoff)
                time.sleep(backoff)
        raise AssertionError("unreachable")

    # -- public API --------------------------------------------------------
    def get_contract_spec(self, symbol: str) -> ContractSpec:
        vs = self.to_venue_symbol(symbol)
        if vs in self._specs:
            return self._specs[vs]
        payload = self._get("/api/v1/contract/detail", {"symbol": vs})
        if not payload.get("success", False):
            raise MexcApiError(f"contract detail failed for {vs}: {payload}")
        d = payload["data"]
        spec = ContractSpec(
            symbol=symbol,
            venue_symbol=vs,
            contract_size=float(d["contractSize"]),
            tick_size=float(d["priceUnit"]),
            min_volume=float(d.get("minVol", 1)),
            volume_step=float(d.get("volUnit", 1)),
        )
        self._specs[vs] = spec
        return spec

    def fetch_candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Download candles with open time in [start, end), paging 2000 at a time."""
        if interval not in MEXC_INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; use one of {list(MEXC_INTERVALS)}")
        vs = self.to_venue_symbol(symbol)
        spec = self.get_contract_spec(symbol)
        step = INTERVAL_SECONDS[interval]
        start_s = int(start.astimezone(timezone.utc).timestamp()) // step * step
        end_s = int(end.astimezone(timezone.utc).timestamp())

        frames: list[pd.DataFrame] = []
        cursor = start_s
        while cursor < end_s:
            window_end = min(cursor + step * MAX_CANDLES_PER_REQUEST, end_s)
            payload = self._get(
                f"/api/v1/contract/kline/{vs}",
                {"interval": MEXC_INTERVALS[interval], "start": cursor, "end": window_end - 1},
            )
            df = parse_kline_response(payload, spec.contract_size)
            if not df.empty:
                frames.append(df)
            log.debug("%s %s %s..%s -> %d candles", vs, interval, cursor, window_end, len(df))
            cursor = window_end

        if not frames:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        lo = pd.Timestamp(start_s, unit="s", tz="UTC")
        hi = pd.Timestamp(end_s, unit="s", tz="UTC")
        out = out[(out["timestamp"] >= lo) & (out["timestamp"] < hi)]
        return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

    def fetch_last_price(self, symbol: str) -> float:
        """Latest traded price: GET /api/v1/contract/ticker?symbol=..."""
        payload = self._get("/api/v1/contract/ticker", {"symbol": self.to_venue_symbol(symbol)})
        if not payload.get("success", False):
            raise MexcApiError(f"ticker failed: {str(payload)[:200]}")
        data = payload["data"]
        if isinstance(data, list):
            data = data[0]
        return float(data["lastPrice"])

    def fetch_funding_history(self, symbol: str, since: datetime, page_size: int = 100,
                              max_pages: int = 200) -> pd.DataFrame:
        """Settled funding rates, newest first from MEXC, returned oldest first.
        GET /api/v1/contract/funding_rate/history?symbol=&page_num=&page_size="""
        vs = self.to_venue_symbol(symbol)
        since_ms = int(since.timestamp() * 1000)
        rows: list[dict] = []
        for page in range(1, max_pages + 1):
            payload = self._get("/api/v1/contract/funding_rate/history",
                                {"symbol": vs, "page_num": page, "page_size": page_size})
            if not payload.get("success", False):
                raise MexcApiError(f"funding history failed: {str(payload)[:200]}")
            data = payload.get("data") or {}
            items = data.get("resultList") if isinstance(data, dict) else data
            if not items:
                break
            for it in items:
                try:
                    rows.append({"time": pd.Timestamp(int(it["settleTime"]), unit="ms", tz="UTC"),
                                 "rate": float(it["fundingRate"])})
                except (KeyError, TypeError, ValueError):
                    continue
            oldest = min(int(it.get("settleTime", since_ms)) for it in items)
            total_pages = data.get("totalPage") if isinstance(data, dict) else None
            if oldest <= since_ms or (total_pages and page >= total_pages):
                break
        if not rows:
            return pd.DataFrame(columns=["time", "rate"])
        df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        return df[df["time"] >= pd.Timestamp(since_ms, unit="ms", tz="UTC")].reset_index(drop=True)

    def close(self) -> None:
        self._client.close()
