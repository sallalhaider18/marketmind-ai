"""Trade-length profiles tested side by side."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str             # short id
    label: str            # shown in reports
    interval: str         # candles the setups are read on
    trend_interval: str   # higher timeframe for the trend filter
    history_days: int     # how much history to test on
    max_hold_bars: int    # close the trade at market after this many bars
    pullback_expiry: int  # bars a pullback limit order stays valid


PROFILES = {
    "scalp": Profile("scalp", "Scalp (minutes, 1m candles)", "1m", "15m", 90, 60, 10),
    "intraday": Profile("intraday", "A few hours (15m candles)", "15m", "1h", 365, 32, 6),
    "swing": Profile("swing", "Swing, 1-3 days (1h candles)", "1h", "4h", 1095, 72, 6),
}

SETUPS = ("pullback", "breakout")
TARGETS_R = (1.0, 1.5, 2.0)   # take-profit distance as a multiple of the stop distance
