import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.adapters.base import Trade
from app.data.live_stream import LiveMarketState
from app.obtrade.engine import PlanEngine, active_sessions, next_check_time
from app.obtrade.levels import Level, find_levels, split_levels
from app.obtrade.planner import find_setup
from app.orderbook.book import OrderBook

CS = 0.0001   # BTC contract size


def zigzag_4h(points, start="2026-07-01"):
    """4h candles moving linearly between turning points, so swing highs/lows sit at known prices."""
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes += list(np.linspace(a, b, 8, endpoint=False))
    closes.append(points[-1])
    closes = np.array(closes)
    ts = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": np.r_[closes[0], closes[:-1]],
                         "high": np.maximum(closes, np.r_[closes[0], closes[:-1]]),
                         "low": np.minimum(closes, np.r_[closes[0], closes[:-1]]), "close": closes})


def test_levels_cluster_repeated_turning_points():
    df = zigzag_4h([80000, 82000, 80000, 82010, 79990, 82000, 81000])
    lv = find_levels(df)
    near_80k = [x for x in lv if abs(x.price - 80000) < 100]
    near_82k = [x for x in lv if abs(x.price - 82000) < 100]
    assert near_80k and near_80k[0].touches >= 2
    assert near_82k and near_82k[0].touches >= 3
    sup, res = split_levels(lv, 81000)
    assert sup[0].price < 81000 < res[0].price


def daily(points, start="2026-03-01"):
    d = zigzag_4h(points, start)
    d["timestamp"] = pd.date_range(start, periods=len(d), freq="1D", tz="UTC")
    return d


def test_daily_turns_count_more_and_weak_levels_are_dropped():
    df4 = zigzag_4h([80000, 81000, 80000, 81500, 80600, 81200])      # many small 4h swings
    df1 = daily([78000, 84000, 78050, 84100, 79000, 81000])          # big daily swings
    lv = find_levels(df4, df1, current=81000)
    prices = [round(x.price, -2) for x in lv]
    assert 78000 in prices and 84000 in prices                     # daily levels kept
    d78 = next(x for x in lv if abs(x.price - 78000) < 200)
    assert d78.daily == 2 and d78.score >= 6


def test_levels_are_spread_out_and_limited():
    pts = [80000 + 300 * (i % 2) + (i // 2) * 600 for i in range(30)]   # lots of nearby turns
    lv = find_levels(zigzag_4h(pts), current=84000)
    ps = sorted(x.price for x in lv)
    assert all((b - a) / a >= 0.008 for a, b in zip(ps, ps[1:]))     # at least 0.8% apart
    assert sum(p < 84000 for p in ps) <= 3 and sum(p > 84000 for p in ps) <= 3


def test_levels_ignore_unconfirmed_last_candles():
    df = zigzag_4h([80000, 82000, 80000, 83000])     # the last high has no candles after it
    assert not any(abs(x.price - 83000) < 50 for x in find_levels(df))


def book_with_wall(mid=81000.0, wall_at=80800.0, wall_contracts=200000, side="bid"):
    b = OrderBook()
    bids = [[round(mid - 5 - 10 * i, 1), 2000, 1] for i in range(100)]      # down to ~80,000
    asks = [[round(mid + 5 + 10 * i, 1), 2000, 1] for i in range(100)]      # up to ~82,000
    target = bids if side == "bid" else asks
    for lv in target:
        if abs(lv[0] - wall_at) <= 5:
            lv[1] = wall_contracts
    b.load_snapshot(bids, asks, 1)
    return b


LEVELS = [Level(80500, 2, ""), Level(80800, 3, ""), Level(81500, 2, ""), Level(81800, 2, "")]


def test_long_setup_at_support_with_big_bid_wall():
    setup, why = find_setup(book_with_wall(), CS, LEVELS)
    assert why == "ok" and setup.side == "long"
    assert setup.level_price == 80800 and setup.level_touches == 3
    assert 80800 <= setup.entry < 81000                     # just in front of the wall, below price
    assert setup.stop == pytest.approx(80500 * 0.999)        # beyond the next support down
    assert setup.target == pytest.approx(81500 * 0.999)      # before the nearest resistance
    assert setup.wall_ratio > 50 and setup.rr > 1.2


def test_short_setup_at_resistance_with_big_ask_wall():
    setup, _ = find_setup(book_with_wall(wall_at=81500, side="ask"), CS, LEVELS)
    assert setup.side == "short" and setup.level_price == 81500
    assert setup.stop == pytest.approx(81800 * 1.001) and setup.target == pytest.approx(80800 * 1.001)


def test_setup_refused_when_reward_too_small():
    tight = [Level(80300, 2, ""), Level(80800, 3, ""), Level(81500, 2, "")]   # stop far, target near
    setup, why = find_setup(book_with_wall(), CS, tight)
    assert setup is None and "reward:risk" in why


def test_no_setup_without_wall_or_beyond_visible_book():
    setup, why = find_setup(book_with_wall(wall_contracts=2000), CS, LEVELS)
    assert setup is None and "no wall" in why
    far = [Level(78000, 3, ""), Level(77000, 2, ""), Level(84000, 2, "")]
    setup, _ = find_setup(book_with_wall(), CS, far)
    assert setup is None


def test_sessions_and_check_schedule():
    t = datetime(2026, 7, 15, 14, 7, tzinfo=timezone.utc)     # 15:07 London, 10:07 NY
    assert set(active_sessions(t)) == {"London", "New York"}
    assert active_sessions(datetime(2026, 7, 15, 2, 0, tzinfo=timezone.utc)) == ["Asia"]   # 11:00 Tokyo
    assert next_check_time(t) == datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc)
    # gap between New York close (21:00 UTC in summer) and Tokyo open (00:00 UTC)
    assert next_check_time(datetime(2026, 7, 15, 21, 10, tzinfo=timezone.utc)) == \
        datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- engine --
class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def make_engine(tmp_path, book):
    live = LiveMarketState(["BTCUSDT.P"], stale_after=1e9, clock=lambda: 0.0)
    live.connected = True
    st = live.symbols["BTCUSDT.P"]
    st.book.book, st.book.synced, st.last_message_at = book, True, 0.0
    clock = Clock(datetime(2026, 7, 15, 14, 0, 5, tzinfo=timezone.utc))
    lines = []
    df4 = zigzag_4h([80300, 81500, 80800, 81900, 80300, 81500, 80800, 81000])
    eng = PlanEngine(live, lambda s: CS, lambda s, iv: df4, tmp_path / "s.json", tmp_path / "t.csv",
                     clock=clock, out=lines.append)
    eng.levels["BTCUSDT.P"] = LEVELS
    return eng, live, clock, lines


def trade(live, price, t):
    live.symbols["BTCUSDT.P"].trades.append(Trade("BTCUSDT.P", t, price, 1.0, "buy"))


def run(coro):
    return asyncio.run(coro)


def test_engine_places_fills_and_hits_target(tmp_path, monkeypatch):
    eng, live, clock, lines = make_engine(tmp_path, book_with_wall())
    monkeypatch.setattr("app.obtrade.engine.find_levels", lambda *a, **k: LEVELS)
    run(eng.step())
    plan = eng.st["active"]["BTCUSDT.P"]
    assert plan["status"] == "pending" and any(">>> PLACE LIMIT BUY" in x for x in lines)
    clock.t += timedelta(minutes=5)
    trade(live, plan["entry"] - 1, clock.t)                 # price trades down to the limit
    run(eng.step())
    assert eng.st["active"]["BTCUSDT.P"]["status"] == "open" and any("FILLED" in x for x in lines)
    clock.t += timedelta(hours=3)
    trade(live, plan["target"] + 5, clock.t)
    run(eng.step())
    assert eng.st["closed"] == 1 and eng.st["wins"] == 1
    assert eng.st["total_r"] == pytest.approx(plan["rr"], rel=0.01)
    assert eng.st["history"][0]["result"] == "target"
    # the wall is still there at the next 30-min check, so a fresh order is placed
    assert eng.st["active"]["BTCUSDT.P"]["status"] == "pending"
    assert pd.read_csv(tmp_path / "t.csv")["reason"].iloc[0] == "target"


def test_engine_cancels_when_wall_is_pulled(tmp_path, monkeypatch):
    eng, live, clock, lines = make_engine(tmp_path, book_with_wall())
    monkeypatch.setattr("app.obtrade.engine.find_levels", lambda *a, **k: LEVELS)
    run(eng.step())
    assert "BTCUSDT.P" in eng.st["active"]
    live.symbols["BTCUSDT.P"].book.book = book_with_wall(wall_contracts=2000)   # wall gone
    clock.t += timedelta(minutes=30)
    run(eng.step())
    assert not eng.st["active"] and any("wall pulled" in x for x in lines)
    assert eng.st["history"][0]["status"] == "cancelled"


def test_engine_stop_and_restart_keeps_state(tmp_path, monkeypatch):
    eng, live, clock, lines = make_engine(tmp_path, book_with_wall())
    monkeypatch.setattr("app.obtrade.engine.find_levels", lambda *a, **k: LEVELS)
    run(eng.step())
    plan = eng.st["active"]["BTCUSDT.P"]
    clock.t += timedelta(minutes=2)
    trade(live, plan["entry"], clock.t)
    run(eng.step())
    eng2, *_ = make_engine(tmp_path, book_with_wall())
    assert eng2.st["active"]["BTCUSDT.P"]["status"] == "open"   # survives a restart
    clock.t += timedelta(minutes=2)
    trade(live, plan["stop"] - 5, clock.t)
    run(eng.step())
    assert eng.st["closed"] == 1 and eng.st["wins"] == 0 and eng.st["total_r"] < -0.99


def test_no_new_plans_outside_sessions(tmp_path, monkeypatch):
    eng, live, clock, lines = make_engine(tmp_path, book_with_wall())
    monkeypatch.setattr("app.obtrade.engine.find_levels", lambda *a, **k: LEVELS)
    clock.t = datetime(2026, 7, 15, 22, 0, 5, tzinfo=timezone.utc)    # 23:00 London, 18:00 NY, 07:00 Tokyo
    run(eng.step())
    assert not eng.st["active"] and lines == []
    v = eng.view("BTCUSDT.P")
    assert v["sessions"] == [] and v["next_check"].startswith("2026-07-16T00:00")


def test_check_retried_when_book_becomes_ready(tmp_path, monkeypatch):
    eng, live, clock, lines = make_engine(tmp_path, book_with_wall())
    monkeypatch.setattr("app.obtrade.engine.find_levels", lambda *a, **k: LEVELS)
    live.symbols["BTCUSDT.P"].book.synced = False
    run(eng.step())
    assert not eng.st["active"] and any("not ready" in x for x in lines)
    live.symbols["BTCUSDT.P"].book.synced = True
    clock.t += timedelta(seconds=10)
    run(eng.step())                                   # same 30-min slot: retried straight away
    assert eng.st["active"]["BTCUSDT.P"]["status"] == "pending"
