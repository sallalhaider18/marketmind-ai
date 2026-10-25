import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import Costs, RunConfig, run_backtest
from app.backtest.metrics import summarize
from app.strategy.indicators import higher_tf_trend
from app.strategy.setups import features, signals

T0 = pd.Timestamp("2026-01-01", tz="UTC")
NO_COSTS = Costs(0, 0, 0, 0)


def bars(ohlc, step_min=15):
    return pd.DataFrame({
        "timestamp": [T0 + pd.Timedelta(minutes=i * step_min) for i in range(len(ohlc))],
        "open": [b[0] for b in ohlc], "high": [b[1] for b in ohlc],
        "low": [b[2] for b in ohlc], "close": [b[3] for b in ohlc],
        "volume": 1.0, "contracts": 1.0, "quote_volume": 1.0,
    })


def sig_at(n, i, side, etype, entry, stop, atr=1.0):
    s = pd.DataFrame({"side": 0, "entry_type": etype, "entry": np.nan, "stop": np.nan, "atr": atr}, index=range(n))
    s.loc[i, ["side", "entry", "stop"]] = [side, entry, stop]
    return s


def run(df, s, **kw):
    cfg = RunConfig(costs=kw.pop("costs", NO_COSTS), **kw)
    return run_backtest(df, s, cfg, 900)


FLAT = (100, 100.2, 99.8, 100)


def test_market_long_hits_target_next_bar():
    df = bars([FLAT, (100, 100.5, 99.5, 100.4), (100.4, 102.5, 100.3, 102.2), FLAT])
    trades, st = run(df, sig_at(4, 0, 1, "market", 100, 99), target_r=2)
    t = trades[0]
    assert t["entry"] == 100 and t["target"] == 102 and t["exit_reason"] == "target"
    assert t["r"] == pytest.approx(2.0) and t["pnl_usd"] == pytest.approx(200)


def test_stop_and_target_in_same_bar_counts_as_loss():
    df = bars([FLAT, (100, 100.5, 99.5, 100), (100, 103, 98, 100), FLAT])
    trades, _ = run(df, sig_at(4, 0, 1, "market", 100, 99), target_r=2)
    assert trades[0]["exit_reason"] == "stop" and trades[0]["r"] == pytest.approx(-1)


def test_gap_through_stop_exits_at_open():
    df = bars([FLAT, (100, 100.5, 99.5, 100), (97, 97.5, 96, 97), FLAT])
    trades, _ = run(df, sig_at(4, 0, 1, "market", 100, 99))
    assert trades[0]["exit"] == 97 and trades[0]["r"] == pytest.approx(-3)


def test_limit_fill_cannot_take_profit_on_fill_bar():
    # limit buy at 99.5; bar 1 trades down to 99.4 then up to 103 (target 101.5)
    df = bars([FLAT, (100, 103, 99.4, 102), (102, 102.2, 101.6, 101.8), (101.8, 101.9, 101.7, 101.8)])
    trades, st = run(df, sig_at(4, 0, 1, "limit", 99.5, 98.5), target_r=2, max_hold_bars=10)
    t = trades[0]
    assert st["limit_filled"] == 1 and t["entry"] == 99.5
    assert t["exit_reason"] != "target" or t["exit_time"] > df["timestamp"][1]


def test_limit_fills_at_better_open():
    df = bars([FLAT, (99.0, 99.2, 98.9, 99.1), (99.1, 102, 99, 101.9), FLAT])
    trades, _ = run(df, sig_at(4, 0, 1, "limit", 99.5, 98.5), target_r=1)
    assert trades[0]["entry"] == 99.0


def test_limit_expires_unfilled():
    df = bars([FLAT] + [(100, 100.4, 99.9, 100.2)] * 8)
    trades, st = run(df, sig_at(9, 0, 1, "limit", 99.0, 98.0), limit_expiry_bars=3)
    assert trades == [] and st["limit_expired"] == 1 and st["limit_filled"] == 0


def test_timeout_exit_and_short_side():
    df = bars([FLAT] + [FLAT] * 6)
    trades, _ = run(df, sig_at(7, 0, -1, "market", 100, 101), max_hold_bars=3)
    t = trades[0]
    assert t["side"] == "short" and t["exit_reason"] == "timeout" and t["bars_held"] == 3


def test_costs_reduce_result():
    df = bars([FLAT, (100, 100.5, 99.5, 100.4), (100.4, 102.5, 100.3, 102.2), FLAT])
    costs = Costs(maker_fee=0.0, taker_fee=0.0002, slippage=0.0001, funding_per_8h=0.0001)
    trades, _ = run(df, sig_at(4, 0, 1, "market", 100, 99), target_r=2, costs=costs)
    t = trades[0]
    assert t["entry"] == pytest.approx(100.01)                       # slippage on entry
    assert 0 < t["costs_usd"] and t["r"] < 2.0


def test_funding_charged_per_8h_crossing():
    ohlc = [FLAT] * 70        # 15m bars: 70 bars = 17.5 h
    df = bars(ohlc)
    costs = Costs(0, 0, 0, funding_per_8h=0.001)
    trades, _ = run(df, sig_at(70, 0, 1, "market", 100, 99), max_hold_bars=40, costs=costs)
    # held 40 bars (10 h) from 00:15 -> crosses 08:00 once
    assert trades[0]["costs_usd"] == pytest.approx(0.001 * 100 * trades[0]["qty"])


def test_leverage_cap_limits_size():
    df = bars([FLAT, (100, 100.5, 99.5, 100), FLAT])
    trades, _ = run(df, sig_at(3, 0, 1, "market", 100, 99.99), min_stop_atr=0)  # 0.01 stop
    assert trades[0]["qty"] * 100 <= 10_000 * 10 + 1e-6


def test_one_position_at_a_time():
    df = bars([FLAT] * 10)
    s = sig_at(10, 0, 1, "market", 100, 99)
    s.loc[1:5, ["side", "entry", "stop"]] = [1, 100, 99]
    trades, st = run(df, s, max_hold_bars=20)
    assert len(trades) == 1   # still in the first trade, later setups ignored


def test_summary_metrics():
    t = pd.DataFrame({"r": [2.0, -1.0, -1.0, 2.0], "pnl_usd": [200, -100, -100, 200], "bars_held": 3,
                      "side": ["long"] * 4, "costs_usd": 1.0, "risk_usd": 100.0})
    m = summarize(t)
    assert m["win_rate"] == 0.5 and m["expectancy_r"] == 0.5 and m["profit_factor"] == 2.0
    assert m["total_return_pct"] == pytest.approx(2.0) and m["max_drawdown_pct"] < 0


# ---------------------------------------------------------- no look-ahead --
def random_walk(n=3000, seed=1, step_min=15):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    df = bars(list(zip(open_, high, low, close)), step_min)
    df["volume"] = rng.uniform(1, 3, n)
    return df


def test_signals_do_not_use_future_data():
    df = random_walk()
    cut = 2000
    f1 = features(df, "15m", "1h")
    altered = df.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.3     # change the future
    altered.loc[cut:, "volume"] *= 5
    f2 = features(altered, "15m", "1h")
    for setup in ("pullback", "breakout"):
        s1, s2 = signals(setup, df, f1), signals(setup, altered, f2)
        pd.testing.assert_series_equal(s1["side"][:cut], s2["side"][:cut])
    pd.testing.assert_frame_equal(f1.iloc[:cut], f2.iloc[:cut])


def test_higher_tf_trend_waits_for_bar_close():
    df = random_walk(400)
    tr = higher_tf_trend(df, "15m", "1h")
    # Changing the last 15m bar of an hour must not affect bars inside that hour
    altered = df.copy()
    i = 7  # 01:45 bar, closes the 01:00 hour
    altered.loc[i, ["close", "high"]] = altered.loc[i, ["close", "high"]] * 2
    tr2 = higher_tf_trend(altered, "15m", "1h")
    pd.testing.assert_frame_equal(tr.iloc[:i], tr2.iloc[:i])


def test_setups_produce_trades_on_random_walk():
    df = random_walk(4000, seed=3)
    f = features(df, "15m", "1h")
    for setup in ("pullback", "breakout"):
        trades, st = run_backtest(df, signals(setup, df, f), RunConfig(max_hold_bars=32), 900)
        assert st["signals"] > 0 and len(trades) > 0
        assert all(t["risk_usd"] > 0 for t in trades)


def test_t_stat_separates_noise_from_edge():
    rng = np.random.default_rng(0)
    base = {"bars_held": 3, "side": "long", "costs_usd": 1.0, "risk_usd": 100.0}
    noise = pd.DataFrame({"r": rng.choice([1.0, -1.0], 300), **base})
    noise["pnl_usd"] = noise["r"] * 100
    edge = pd.DataFrame({"r": rng.choice([1.0, -1.0], 300, p=[0.62, 0.38]), **base})
    edge["pnl_usd"] = edge["r"] * 100
    assert abs(summarize(noise)["t_stat"]) < 2
    assert summarize(edge)["t_stat"] > 2
