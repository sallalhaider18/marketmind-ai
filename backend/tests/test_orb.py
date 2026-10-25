import pandas as pd
import pytest

from app.backtest.engine import Costs
from app.strategy.orb import OrbConfig, run_orb, session_days

NO_COSTS = Costs(0, 0, 0, 0)


def ny_day(date, bars_after_or, or_bars=None, step=5):
    """Candles from 9:30 ET on `date`. or_bars: list of (o,h,l,c) for 9:30-9:45;
    bars_after_or: list of (o,h,l,c) from 9:45 on."""
    or_bars = or_bars or [(100, 101, 99.5, 100.5), (100.5, 102, 100, 101), (101, 101.5, 99, 100)]
    allb = or_bars + bars_after_or
    start = pd.Timestamp(f"{date} 09:30", tz="America/New_York").tz_convert("UTC")
    return pd.DataFrame({
        "timestamp": [start + pd.Timedelta(minutes=i * step) for i in range(len(allb))],
        "open": [b[0] for b in allb], "high": [b[1] for b in allb], "low": [b[2] for b in allb],
        "close": [b[3] for b in allb], "volume": 1.0, "contracts": 1.0, "quote_volume": 1.0,
    })
# opening range from the default OR bars: high 102, low 99, range 3


def run(df, **kw):
    return run_orb(df, 5, OrbConfig(costs=NO_COSTS, **kw))


def test_opening_range_uses_new_york_time_in_summer_and_winter():
    for date, utc_hour in (("2026-07-15", 13), ("2026-01-15", 14)):   # EDT vs EST
        df = ny_day(date, [(100, 100.5, 99.5, 100)] * 4)
        assert df["timestamp"].iloc[0].hour == utc_hour
        days = list(session_days(df, 5))
        assert len(days) == 1


def test_weekends_skipped():
    df = ny_day("2026-07-18", [(100, 100.5, 99.5, 100)] * 4)   # Saturday
    assert list(session_days(df, 5)) == []


def test_close_entry_long_hits_1x_target():
    after = [(101, 102.6, 100.8, 102.5),   # 9:45 closes above 102 -> breakout
             (102.5, 103, 102.2, 102.8),   # entry at open 102.5
             (102.8, 105.2, 102.7, 105)]   # 1x target = 102 + 3 = 105
    trades, st = run(ny_day("2026-07-15", after), entry="close", stop="range", target="1x")
    t = trades[0]
    assert t["side"] == "long" and t["entry"] == 102.5 and t["stop"] == 99 and t["exit_reason"] == "target"
    assert t["r"] == pytest.approx((105 - 102.5) / (102.5 - 99))


def test_retest_entry_fills_at_edge_and_mid_stop():
    after = [(101, 102.8, 100.8, 102.6),   # breakout close
             (102.6, 103.5, 102.4, 103.2),  # no retest yet
             (103.2, 103.3, 101.9, 102.3),  # retest: fills at 102
             (102.3, 104.0, 102.2, 103.8),
             (103.8, 105.5, 103.7, 105.2)]  # target 105
    trades, _ = run(ny_day("2026-07-15", after), entry="retest", stop="mid", target="1x")
    t = trades[0]
    assert t["entry"] == 102 and t["stop"] == 100.5 and t["exit_reason"] == "target"
    assert t["r"] == pytest.approx(3 / 1.5)


def test_retest_cancelled_if_target_reached_first():
    after = [(101, 102.8, 100.8, 102.6), (102.6, 105.5, 102.5, 105.2), (105, 105.5, 101.5, 102)]
    trades, st = run(ny_day("2026-07-15", after), entry="retest", target="1x")
    assert trades == [] and st["retest_missed"] == 1


def test_partial_then_breakeven():
    after = [(101, 102.6, 100.8, 102.5),
             (102.5, 103, 102.2, 102.8),     # entry 102.5
             (102.8, 105.3, 102.9, 105.1),   # tp1 105: half off, stop -> 102.5
             (105.1, 105.2, 102.4, 102.6)]   # back to entry: rest at break-even
    trades, _ = run(ny_day("2026-07-15", after), entry="close", stop="range", target="partial")
    t = trades[0]
    assert t["exit_reason"] == "breakeven"
    assert t["pnl_usd"] == pytest.approx(t["qty"] / 2 * (105 - 102.5))


def test_stop_and_target_same_candle_is_a_loss():
    after = [(101, 102.6, 100.8, 102.5), (102.5, 106, 98, 102)]
    trades, _ = run(ny_day("2026-07-15", after), entry="close", stop="range", target="1x")
    assert trades[0]["exit_reason"] == "stop" and trades[0]["r"] == pytest.approx(-1)


def test_short_breakout_and_session_close():
    after = [(100, 100.2, 98.5, 98.7)] + [(98.7, 98.9, 98.4, 98.6)] * 74   # 9:50 .. 15:55
    trades, _ = run(ny_day("2026-07-15", after), entry="close", stop="range", target="2x")
    t = trades[0]
    assert t["side"] == "short" and t["exit_reason"] == "session close"
    assert t["exit_time"].tz_convert("America/New_York").hour == 16


def test_no_breakout_after_11am():
    after = [(100, 101, 99.5, 100)] * 15 + [(101, 103, 100.9, 102.8)] * 3   # breakout at 11:00+
    trades, st = run(ny_day("2026-07-15", after), entry="close")
    assert trades == [] and st["breakouts"] == 0


def test_day_with_missing_opening_candle_is_skipped():
    df = ny_day("2026-07-15", [(101, 102.6, 100.8, 102.5)] * 3).drop(index=1)
    assert list(session_days(df, 5)) == []


@pytest.mark.parametrize("target,k", [("1:1", 1.0), ("1.5:1", 1.5), ("2:1", 2.0)])
def test_reward_to_risk_targets_use_stop_distance(target, k):
    # entry 102.5, range stop 99 -> risk 3.5; target = 102.5 + k * 3.5
    after = [(101, 102.6, 100.8, 102.5), (102.5, 103, 102.2, 102.8)] + \
            [(102.8, 102.5 + k * 3.5 + 0.1, 102.7, 102.9)]
    trades, _ = run(ny_day("2026-07-15", after), entry="close", stop="range", target=target)
    t = trades[0]
    assert t["target"] == pytest.approx(102.5 + k * 3.5)
    assert t["exit_reason"] == "target" and t["r"] == pytest.approx(k)


# ------------------------------------------------- fixed distance / holding --
def btc_day(date, after, step=5):
    """BTC-like prices: opening range 82,000-82,300."""
    or_bars = [(82100, 82300, 82000, 82200), (82200, 82250, 82050, 82150), (82150, 82280, 82100, 82250)]
    return ny_day(date, after, or_bars=or_bars, step=step)


def test_fixed_1000_stop_and_target_on_btc():
    after = [(82250, 82400, 82240, 82350),   # 9:45 closes above 82,300 -> long
             (82350, 82500, 82300, 82450),   # entry at open 82,350
             (82450, 83400, 82400, 83300)]   # target 83,350 hit
    trades, _ = run(btc_day("2026-07-15", after), entry="close", fixed_distance=1000)
    t = trades[0]
    assert t["entry"] == 82350 and t["stop"] == 81350 and t["target"] == 83350
    assert t["exit_reason"] == "target" and t["r"] == pytest.approx(1.0)


def test_fixed_pct_scales_with_price():
    after = [(82250, 82400, 82240, 82350), (82350, 82500, 82300, 82450)] + [(82450, 82500, 82400, 82450)] * 3
    trades, _ = run(btc_day("2026-07-15", after), entry="close", fixed_pct=0.01)
    assert trades[0]["stop"] == pytest.approx(82350 * 0.99)


def test_hold_past_session_close_until_target():
    # entry ~9:50 ET, nothing happens until after 16:00, target hit next morning
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 120 + \
            [(82400, 83500, 82390, 83400)] + [(83400, 83450, 83350, 83400)] * 10
    df = btc_day("2026-07-15", after)
    day_trade, _ = run(df, entry="close", fixed_distance=1000)
    held, _ = run(df, entry="close", fixed_distance=1000, hold_hours=48)
    assert day_trade[0]["exit_reason"] == "session close"
    assert held[0]["exit_reason"] == "target" and held[0]["exit_time"] > day_trade[0]["exit_time"]


def test_no_new_trade_while_previous_still_open():
    def day(date):
        return btc_day(date, [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 74)
    df = pd.concat([day("2026-07-15"), day("2026-07-16")], ignore_index=True)
    one, st = run(df, entry="close", fixed_distance=1000, hold_hours=48)
    two, _ = run(df, entry="close", fixed_distance=1000)
    assert len(one) == 1 and st["skipped_open_trade"] == 1
    assert len(two) == 2


def test_exit_just_before_next_new_york_open():
    # Wednesday trade, nothing hit: closed by the candle that closes at 9:30 ET Thursday
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 400
    trades, _ = run(btc_day("2026-07-15", after), entry="close", fixed_distance=1000, exit_next_open=True)
    t = trades[0]
    out = t["exit_time"].tz_convert("America/New_York")
    assert t["exit_reason"] == "next-day close (9:29 ET)"
    assert (out.day, out.hour, out.minute) == (16, 9, 30)


def test_weekdays_only_friday_trade_held_to_monday():
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 1200
    trades, _ = run(btc_day("2026-07-17", after), entry="close", fixed_distance=1000, exit_next_open=True)
    out = trades[0]["exit_time"].tz_convert("America/New_York")
    assert out.weekday() == 0 and (out.hour, out.minute) == (9, 30)


def test_target_overnight_before_next_open():
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 150 + \
            [(82400, 83500, 82390, 83400)] + [(83400, 83450, 83350, 83400)] * 200
    trades, _ = run(btc_day("2026-07-15", after), entry="close", fixed_distance=1000, exit_next_open=True)
    t = trades[0]
    assert t["exit_reason"] == "target" and t["r"] == pytest.approx(1.0)
    assert t["exit_time"].tz_convert("America/New_York").day == 15   # ~22:30 ET same evening


def test_hold_until_next_day_929_and_weekends():
    # Saturday 2026-07-18: entry after 9:45, nothing hit, closes on the candle ending 9:30 Sunday
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 300
    df = btc_day("2026-07-18", after)
    assert run(df, entry="close", fixed_distance=1000, exit_next_open=True)[0] == []      # weekdays only
    trades, _ = run(df, entry="close", fixed_distance=1000, exit_next_open=True, weekends=True)
    t = trades[0]
    ny_exit = t["exit_time"].tz_convert("America/New_York")
    assert t["exit_reason"] == "next-day close (9:29 ET)"
    assert (ny_exit.day, ny_exit.hour, ny_exit.minute) == (19, 9, 30)   # candle 9:25-9:30 on Sunday


def test_with_weekends_friday_trade_closes_saturday():
    after = [(82250, 82400, 82240, 82350)] + [(82350, 82450, 82300, 82400)] * 1200
    trades, _ = run(btc_day("2026-07-17", after), entry="close", fixed_distance=1000, exit_next_open=True,
                    weekends=True)
    out = trades[0]["exit_time"].tz_convert("America/New_York")
    assert out.weekday() == 5 and (out.hour, out.minute) == (9, 30)


def test_fixed_stop_500_target_1000():
    after = [(82250, 82400, 82240, 82350), (82350, 82500, 82300, 82450),
             (82450, 83400, 82400, 83300)]   # target 82,350 + 1,000 = 83,350 hit
    trades, _ = run(btc_day("2026-07-15", after), entry="close", fixed_distance=500, fixed_rr=2.0)
    t = trades[0]
    assert t["stop"] == 81850 and t["target"] == 83350 and t["r"] == pytest.approx(2.0)
