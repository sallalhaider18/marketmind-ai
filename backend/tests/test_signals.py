import pytest

from app.signals.engine import SignalEngine
from app.signals.rows import RowRecorder, build_row, load_rows
from app.signals.rules import Rules
from app.signals.sim import Position, close_position, open_position, plan_rr, summarize

T0 = 1_800_000_000 - (1_800_000_000 % 1800)      # aligned to a 30-min boundary
SUP, RES = 100.0, 103.0


def row(t, mid, buy=1.0, sell=0.2, imb=0.4, wall=80.0, wall_age=120.0, side_wall="bid", lo=None, hi=None,
        valid=True, wall_price=None):
    w = [{"price": wall_price or (SUP if side_wall == "bid" else RES), "coins": wall, "peak": wall,
          "age": wall_age}] if wall else []
    if mid is None:
        return {"t": t, "valid": False, "mid": None}
    return {"t": t, "valid": valid, "mid": mid, "bid": mid - 0.01, "ask": mid + 0.01, "last": mid,
            "hi": hi if hi is not None else mid, "lo": lo if lo is not None else mid,
            "buy": buy, "sell": sell, "imb10": imb, "imb25": imb,
            "walls_bid": w if side_wall == "bid" else [], "walls_ask": w if side_wall == "ask" else []}


def engine(levels=(SUP, RES), **rules):
    events = []
    e = SignalEngine("BTCUSDT.P", "BTC", lambda t: list(levels), Rules(**rules),
                     on_event=lambda k, d: events.append((k, d)))
    return e, events


def feed(e, start, prices, **kw):
    """One row per second; prices = list of mids."""
    for i, p in enumerate(prices):
        e.on_row(row(start + i, p, **kw))
    return start + len(prices)


def long_setup(e, **kw):
    """Price sits near support, touches it at T0+200, recovers; the 5-min candle closes above at T0+300."""
    t = feed(e, T0, [100.12] * 200, **kw)          # armed (within 0.15%)
    t = feed(e, t, [100.02] * 10, **kw)            # touched
    t = feed(e, t, [100.2] * 95, **kw)             # 5-min candle T0..T0+300 closes at 100.2
    return t


def kinds(events, k):
    return [d for kk, d in events if kk == k]


def test_full_long_signal_opens_a_and_b():
    e, ev = engine()
    long_setup(e)
    sig = kinds(ev, "signal")
    assert len(sig) == 1 and sig[0]["side"] == "long" and sig[0]["level"] == SUP
    assert sig[0]["stop"] == pytest.approx(100.0 * 0.999)
    assert sig[0]["target"] == pytest.approx(103.0 * 0.999)
    evals = kinds(ev, "evaluation")
    assert evals[-1]["passed"] and all(evals[-1][f"ok_{k}"] for k in
                                       ("wall", "imbalance", "flow", "cvd", "price", "risk"))
    a = [p for p in e.positions if p.strategy == "A"][0]
    b = [p for p in e.positions if p.strategy == "B"][0]
    assert a.status == "open" and a.entry == pytest.approx(100.21 * 1.0001)
    assert b.status == "pending" and b.entry == pytest.approx(100.05)
    assert a.risk_usd == pytest.approx(100.0)          # 1% of $10k


def test_both_strategies_reach_target():
    e, ev = engine()
    t = long_setup(e)
    t = feed(e, t, [100.04] * 5)                       # retest: trades below the 100.05 limit -> B filled
    assert kinds(ev, "fill")
    feed(e, t, [101.0, 102.0, 102.95])                 # through the target
    closes = kinds(ev, "close")
    assert {c["strategy"] for c in closes} == {"A", "B"}
    assert all(c["result"] == "target" and c["r"] > 0 for c in closes)
    st = e.stats()
    assert st["A"]["trades"] == 1 and st["B"]["trades"] == 1 and st["B"]["fill_rate"] == 1.0
    assert st["A"]["equity"] > 10_000


def test_b_missed_when_price_runs_without_retest():
    e, ev = engine()
    t = long_setup(e)
    feed(e, t, [101.0, 102.0, 102.95])
    missed = kinds(ev, "missed")
    assert len(missed) == 1 and "target before the limit" in missed[0]["result"]
    assert e.stats()["B"]["missed"] == 1 and e.stats()["B"]["fill_rate"] == 0.0


def test_stop_loss_is_about_minus_one_r():
    e, ev = engine()
    t = long_setup(e)
    feed(e, t, [100.1, 99.89])
    a = [c for c in kinds(ev, "close") if c["strategy"] == "A"][0]
    assert a["result"] == "stop"
    assert -1.4 < a["r"] < -1.0                       # 1 R plus fees, slippage and the small gap


@pytest.mark.parametrize("bad, missing", [
    ({"buy": 0.2, "sell": 1.0}, "flow"),             # sellers in control
    ({"imb": 0.1}, "imbalance"),
    ({"wall": 20.0}, "wall"),                         # wall below the 50 BTC minimum
    ({"wall_age": 30.0}, "wall"),                     # too new
])
def test_missing_condition_is_logged_not_traded(bad, missing):
    e, ev = engine()
    long_setup(e, **bad)
    assert not kinds(ev, "signal") and not e.positions
    last = kinds(ev, "evaluation")[-1]
    assert not last["passed"] and missing in last["reason"] and last[f"ok_{missing}"] is False


def test_shrunk_wall_fails():
    e, ev = engine()
    t = feed(e, T0, [100.12] * 200)
    t = feed(e, t, [100.02] * 10)
    for i in range(95):
        r = row(t + i, 100.2)
        r["walls_bid"] = [{"price": SUP, "coins": 60.0, "peak": 200.0, "age": 300.0}]   # shrunk 70%
        e.on_row(r)
    assert not kinds(ev, "signal") and "wall" in kinds(ev, "evaluation")[-1]["reason"]


def test_no_target_level_rejects_risk():
    e, ev = engine(levels=(SUP,))
    long_setup(e)
    last = kinds(ev, "evaluation")[-1]
    assert not last["passed"] and last["ok_risk"] is False


def test_reward_risk_checked_per_strategy():
    e, ev = engine(levels=(SUP, 100.6))               # close resistance: market entry not worth it, retest is
    long_setup(e)
    assert kinds(ev, "signal")
    assert [d["strategy"] for d in kinds(ev, "skip")] == ["A"]
    assert [p.strategy for p in e.positions] == ["B"]


def test_poor_reward_risk_rejected():
    e, ev = engine(levels=(SUP, 100.38))              # too close for both
    long_setup(e)
    assert not kinds(ev, "signal") and kinds(ev, "evaluation")[-1]["ok_risk"] is False


def test_stale_data_drops_setup_and_logs():
    e, ev = engine()
    t = feed(e, T0, [100.12] * 200)
    t = feed(e, t, [100.02] * 10)
    e.on_row(row(t, None, valid=False))
    assert not e.setups
    assert "data invalid" in kinds(ev, "evaluation")[-1]["reason"]


def test_30min_close_below_level_cancels():
    e, ev = engine()
    t = feed(e, T0, [100.12] * 200)
    t = feed(e, t, [100.02] * 10)
    # stays just below the level: 5-min closes fail "price", then the 30-min candle closes below
    feed(e, t, [99.95] * (1800 - 210 + 2), buy=0.2, sell=1.0)
    reasons = [d["reason"] for d in kinds(ev, "evaluation")]
    assert any("30-min candle closed through" in r for r in reasons)
    assert not kinds(ev, "signal")


def test_short_signal_mirror():
    e, ev = engine()
    kw = dict(buy=0.2, sell=1.0, imb=-0.4, side_wall="ask")
    t = feed(e, T0, [102.88] * 200, **kw)
    t = feed(e, t, [102.98] * 10, **kw)
    feed(e, t, [102.8] * 95, **kw)
    sig = kinds(ev, "signal")
    assert len(sig) == 1 and sig[0]["side"] == "short"
    assert sig[0]["stop"] == pytest.approx(103.0 * 1.001) and sig[0]["target"] == pytest.approx(100.0 * 1.001)


def test_one_trade_per_coin_and_level_pause():
    e, ev = engine()
    t = long_setup(e)
    t = feed(e, t, [100.12] * 300)                     # back near the level while trades are open
    assert len(kinds(ev, "signal")) == 1


def test_plan_rr_counts_costs():
    r = Rules()
    clean = plan_rr("long", 100, 99, 102, 0.0, Rules(taker=0, slip=0))
    assert clean == pytest.approx(2.0)
    assert plan_rr("long", 100, 99, 102, r.taker, r) < clean
    assert plan_rr("long", 100, 99, 99.5, 0, r) is None


def test_position_pnl_and_summary():
    r = Rules()
    p = Position("A", "BTCUSDT.P", "long", 100, 100, 99, 102, 0)
    open_position(p, 10_000, r, 0, r.taker)
    close_position(p, 102, r.maker, 10, "target")
    assert p.qty == pytest.approx(100) and p.pnl == pytest.approx(200 - 100 * 100 * 0.0002)
    s = summarize([{"r": 2, "pnl": 200, "fees": 2}, {"r": -1, "pnl": -100, "fees": 2},
                   {"r": -1, "pnl": -100, "fees": 2}], missed=1)
    assert s["win_rate"] == pytest.approx(1 / 3) and s["expectancy_r"] == pytest.approx(0)
    assert s["profit_factor"] == pytest.approx(1.0) and s["max_drawdown_r"] == pytest.approx(2)
    assert s["fill_rate"] == pytest.approx(0.75)


def test_rows_round_trip(tmp_path):
    rec = RowRecorder(tmp_path)
    rows = [row(T0 + i, 100 + i * 0.01) for i in range(5)]
    for r in rows:
        rec.add("BTCUSDT.P", r)
    rec.flush_all()
    back = load_rows(tmp_path, "BTCUSDT.P")
    assert len(back) == 5 and back[2]["walls_bid"][0]["coins"] == 80.0
    e1, ev1 = engine()
    e2, ev2 = engine()
    for r in rows:
        e1.on_row(r)
    for r in back:
        e2.on_row(r)
    assert e1.view()["live"] == e2.view()["live"]


def test_build_row_from_book():
    from app.orderbook.book import OrderBook
    from app.orderbook.features import book_metrics
    from app.orderbook.large_orders import TrackedOrder
    from app.adapters.base import Trade
    from datetime import datetime, timezone
    b = OrderBook()
    b.load_snapshot([[99.9, 10, 1], [99.5, 500, 1]], [[100.1, 10, 1]], 1)
    m = book_metrics(b, 0.1)
    o = TrackedOrder("bid", 99.5, first_seen=0, last_seen=70, initial=500, size=500, max_size=600)
    now = datetime.now(timezone.utc)
    trades = [Trade("X", now, 100.1, 30, "buy"), Trade("X", now, 99.9, 10, "sell")]
    r = build_row(T0, True, m, trades, 0.1, [o], 70)
    assert r["valid"] and r["buy"] == pytest.approx(3) and r["sell"] == pytest.approx(1)
    assert r["hi"] == 100.1 and r["lo"] == 99.9 and r["last"] == 99.9
    assert r["walls_bid"] == [{"price": 99.5, "coins": 50.0, "peak": 60.0, "age": 70, "hit": 0.0}]
    assert not build_row(T0, False, None, [], 0.1, [], 0)["valid"]


def test_replay_run_and_levels_use_only_closed_candles():
    import pandas as pd
    from scripts.replay_signals import level_source, run
    ts = pd.date_range("2027-01-01", periods=60, freq="4h", tz="UTC")
    df4 = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    df4.loc[30, "low"] = 90.0                                   # a clear 4h swing low
    df4.loc[45, "high"] = 120.0                                 # a later swing high
    df1 = df4.iloc[0:0]
    from app.obtrade.levels import find_levels
    lv = level_source(df4, df1)
    t_mid = ts[40].timestamp() + 60                             # candle 40 still open
    expected = [x.price for x in find_levels(df4.iloc[:40], df1, current=100.0)]
    assert lv(t_mid, 100.0) == expected                         # no look-ahead
    assert 120.0 not in lv(t_mid, 100.0)

    rows = []
    e, _ = engine()
    for i, p in enumerate([100.12] * 200 + [100.02] * 10 + [100.2] * 95 + [101.0, 102.0, 102.95]):
        rows.append(row(T0 + i, p))
    st = run("BTCUSDT.P", rows, lambda t, m: [SUP, RES], Rules())
    assert st["signals"] == 1 and st["A"]["trades"] == 1 and st["B"]["missed"] == 1
    assert st["pass_rates"]["wall"] == 1.0



# ------------------------------------------------- break vs reversal --
def breakdown_rows(e, start=T0):
    """Support 100 gives way: wall pulled, sellers in control, two 5-min closes below 99.9."""
    kw = dict(buy=0.2, sell=1.0, imb=-0.4)
    t = feed(e, start, [100.12] * 200, **kw)                 # armed, bid wall present
    t = feed(e, t, [100.02] * 10, **kw)                      # touched
    t = feed(e, t, [99.85] * 90, wall=0, **kw)               # wall gone (pulled), close 1 below
    t = feed(e, t, [99.8] * 305, wall=0, **kw)               # close 2 below -> breakdown confirmed
    return t


def test_support_breakdown_goes_short():
    e, ev = engine(levels=(97.0, SUP, RES))
    breakdown_rows(e)
    sig = kinds(ev, "signal")
    assert len(sig) == 1 and sig[0]["setup"] == "support breakdown" and sig[0]["side"] == "short"
    assert sig[0]["stop"] == pytest.approx(100.0 * 1.001) and sig[0]["target"] == pytest.approx(97.0 * 1.001)
    assert sig[0]["wall_fate"] == "pulled"
    assert all(p.setup == "support breakdown" for p in e.positions)
    first = [d for d in kinds(ev, "evaluation") if d["candidate"] == "uncertain"]
    assert first, "the first close below is not yet a confirmed break"


def test_breakdown_blocked_while_wall_still_defends():
    e, ev = engine(levels=(97.0, SUP, RES))
    kw = dict(buy=0.2, sell=1.0, imb=-0.4)
    t = feed(e, T0, [100.12] * 200, **kw)
    t = feed(e, t, [100.02] * 10, **kw)
    feed(e, t, [99.85] * 400, wall_price=99.95, **kw)        # big bid still sitting on the level
    assert not kinds(ev, "signal")
    assert any("breakdown candidate" in d["reason"] and "wall" in d["reason"] for d in kinds(ev, "evaluation"))


def test_resistance_breakout_goes_long():
    e, ev = engine(levels=(SUP, RES, 106.0))
    t = feed(e, T0, [102.88] * 200, side_wall="ask")
    t = feed(e, t, [102.98] * 10, side_wall="ask")
    t = feed(e, t, [103.15] * 90, wall=0)
    feed(e, t, [103.2] * 305, wall=0)
    sig = kinds(ev, "signal")
    assert len(sig) == 1 and sig[0]["setup"] == "resistance breakout" and sig[0]["side"] == "long"
    assert sig[0]["stop"] == pytest.approx(103.0 * 0.999) and sig[0]["target"] == pytest.approx(106.0 * 0.999)


def test_wall_shrinking_by_trades_still_counts():
    e, ev = engine()
    t = feed(e, T0, [100.12] * 200)
    t = feed(e, t, [100.02] * 10)
    for i in range(95):
        r = row(t + i, 100.2)
        r["walls_bid"] = [{"price": SUP, "coins": 60.0, "peak": 200.0, "age": 300.0, "hit": 120.0}]
        e.on_row(r)
    sig = kinds(ev, "signal")
    assert len(sig) == 1 and sig[0]["setup"] == "support bounce" and sig[0]["wall_fate"] == "absorbing"


def test_level_test_outcomes_are_recorded():
    e, ev = engine(levels=(97.0, SUP, RES))
    t = breakdown_rows(e)
    feed(e, t, [99.5] * 3400, wall=0, buy=0.2, sell=1.0, imb=-0.4)
    lt = kinds(ev, "level_test")
    assert len(lt) == 1
    x = lt[0]
    assert x["outcome"] == "broke" and x["kind"] == "support" and x["wall_fate"] == "pulled"
    assert x["max_beyond_pct"] == pytest.approx(0.5) and "breakdown" in x["candidates"]
    assert "support breakdown" in x["trades"] and x["minutes_to_outcome"] is not None


def test_level_test_held_and_unresolved():
    e, ev = engine()
    t = long_setup(e)
    feed(e, t, [100.35] * 3600)                               # +0.35% away from support
    assert kinds(ev, "level_test")[0]["outcome"] == "held"
    e2, ev2 = engine()
    t = feed(e2, T0, [100.12] * 200, buy=0.5, sell=0.5, imb=0)
    feed(e2, t, [100.02] * 3700, buy=0.5, sell=0.5, imb=0)
    assert not kinds(ev2, "signal")
    assert kinds(ev2, "level_test")[0]["outcome"] == "unresolved"


def test_results_split_by_setup_type():
    e, ev = engine()
    t = long_setup(e)
    feed(e, t, [101.0, 102.0, 102.95])
    by = e.stats()["by_setup"]
    assert by["support bounce"]["A"]["trades"] == 1
    assert by["support breakdown"]["A"]["trades"] == 0
    assert kinds(ev, "close")[0]["setup"] == "support bounce"
