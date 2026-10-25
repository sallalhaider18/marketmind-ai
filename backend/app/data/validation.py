"""Data-quality checks for candle data.

Nothing downstream (features, model, signals) should run on data that fails these
checks without the failure being visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from app.adapters.base import INTERVAL_SECONDS


@dataclass
class ValidationReport:
    rows: int
    first: str | None
    last: str | None
    expected_rows: int
    missing_candles: int
    gaps: list[dict] = field(default_factory=list)   # largest gaps first, max 20
    duplicate_timestamps: int = 0
    unsorted: bool = False
    misaligned_timestamps: int = 0                    # not on an interval boundary
    bad_ohlc: int = 0                                 # high/low inconsistent with open/close
    non_positive_prices: int = 0
    negative_volume: int = 0
    zero_volume: int = 0                              # allowed, but worth knowing
    nan_values: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.missing_candles == 0
            and self.duplicate_timestamps == 0
            and not self.unsorted
            and self.misaligned_timestamps == 0
            and self.bad_ohlc == 0
            and self.non_positive_prices == 0
            and self.negative_volume == 0
            and self.nan_values == 0
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d

    def summary(self) -> str:
        if self.rows == 0:
            return "EMPTY: no candles"
        status = "OK" if self.ok else "PROBLEMS FOUND"
        lines = [
            f"{status}: {self.rows} candles from {self.first} to {self.last}",
            f"  missing candles: {self.missing_candles} (in {len(self.gaps)}+ gaps)",
            f"  duplicates: {self.duplicate_timestamps}, unsorted: {self.unsorted}, "
            f"misaligned: {self.misaligned_timestamps}",
            f"  bad OHLC: {self.bad_ohlc}, non-positive prices: {self.non_positive_prices}, "
            f"negative volume: {self.negative_volume}, NaN values: {self.nan_values}",
            f"  zero-volume candles: {self.zero_volume}",
        ]
        for g in self.gaps[:5]:
            lines.append(f"  gap: {g['after']} -> {g['before']} ({g['missing']} missing)")
        return "\n".join(lines)


def validate_candles(df: pd.DataFrame, interval: str) -> ValidationReport:
    step = INTERVAL_SECONDS[interval]
    if df.empty:
        return ValidationReport(rows=0, first=None, last=None, expected_rows=0, missing_candles=0)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    unsorted = not ts.is_monotonic_increasing
    dupes = int(ts.duplicated().sum())

    # Seconds since epoch, independent of pandas' internal datetime resolution
    epoch = (ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    misaligned = int((epoch % step != 0).sum())

    uniq = ts.drop_duplicates().sort_values().reset_index(drop=True)
    first, last = uniq.iloc[0], uniq.iloc[-1]
    expected = int((last - first).total_seconds() // step) + 1

    diffs = uniq.diff().dt.total_seconds().iloc[1:]
    gap_idx = diffs[diffs > step].index
    gaps = [
        {
            "after": uniq.iloc[i - 1].isoformat(),
            "before": uniq.iloc[i].isoformat(),
            "missing": int(diffs.loc[i] // step) - 1,
        }
        for i in gap_idx
    ]
    gaps.sort(key=lambda g: g["missing"], reverse=True)
    missing = sum(g["missing"] for g in gaps)

    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    bad_ohlc = int(((h < o) | (h < c) | (l > o) | (l > c) | (h < l)).sum())
    non_pos = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
    vol = df["volume"].astype(float)

    return ValidationReport(
        rows=len(df),
        first=first.isoformat(),
        last=last.isoformat(),
        expected_rows=expected,
        missing_candles=missing,
        gaps=gaps[:20],
        duplicate_timestamps=dupes,
        unsorted=unsorted,
        misaligned_timestamps=misaligned,
        bad_ohlc=bad_ohlc,
        non_positive_prices=non_pos,
        negative_volume=int((vol < 0).sum()),
        zero_volume=int((vol == 0).sum()),
        nan_values=int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
    )
