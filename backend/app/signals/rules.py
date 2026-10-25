"""All signal-engine numbers in one place.

These are a starting hypothesis, NOT tuned values. The replay script
(`python -m scripts.replay_signals`) tests them on recorded data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rules:
    # 1. Level: price within this distance of a strong daily/4H level
    near_pct: float = 0.0015            # 0.15%
    touch_pct: float = 0.0005           # "touched" = traded within 0.05% of the level
    give_up_candles: int = 12           # stop evaluating 12 x 5-min closes (60 min) after the touch
    break_buf: float = 0.001            # a break needs a 5-min close 0.1% through the level
    # Level tests (labelled outcomes for later study)
    outcome_pct: float = 0.003          # "broke" / "held" = 0.3% beyond / back away first ...
    outcome_horizon: float = 3600       # ... within 60 min of the touch, else "unresolved"
    # 2. Wall: a large order on the level that has persisted
    wall_near_pct: float = 0.0015       # wall within 0.15% of the level
    wall_min_age: float = 60            # seconds
    wall_max_shrink: float = 0.5        # may shrink at most 50% from its peak, unless it shrank by being traded into
    wall_min: dict = field(default_factory=lambda: {"BTC": 50.0, "ETH": 500.0})   # coins
    # 3. Imbalance: average over the window, in BOTH the 0.1% and 0.25% ranges
    imb_min: float = 0.25
    imb_window: float = 60
    imb_coverage: float = 0.9           # need data for 90% of the window
    # 4. Trade flow + CVD
    flow_share: float = 0.60            # aggressive buy (sell) share over the window
    flow_window: float = 60
    # 5. Price reaction: a completed 5-min candle closes back on the right side;
    #    a completed 30-min candle closing through the level cancels the setup
    confirm_tf: int = 300
    invalidate_tf: int = 1800
    # 6. Risk
    rr_min: float = 1.5                 # after fees and slippage
    stop_buf: float = 0.001             # stop 0.1% beyond the level / reaction extreme
    target_buf: float = 0.001           # target 0.1% before the next opposite level
    # Strategy B: limit order at the retest
    retest_pct: float = 0.0005          # limit 0.05% in front of the level
    limit_expiry: float = 3600          # cancel if not filled in 60 min
    max_hold: float = 86400             # day trade: close after 24 h at most
    # Costs (MEXC futures)
    taker: float = 0.0002
    maker: float = 0.0
    slip: float = 0.0001
    # Paper accounts
    start_equity: float = 10_000.0
    risk_pct: float = 0.01

    def wall_min_for(self, asset: str) -> float:
        return float(self.wall_min.get(asset, 0.0))
