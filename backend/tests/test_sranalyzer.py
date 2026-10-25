from datetime import datetime, timezone

import pandas as pd
import pytest

from app.adapters.base import Trade
from app.sranalyzer.analyzer import LevelStore, SRAnalyzer
from app.sranalyzer.measure import AnalyzerRules, classify, second_record, verdict, window

R = AnalyzerRules()
NOW = datetime.now(timezone.utc)
L = 100.0


def buy(p, q):
    return Trade("X", NOW, p, q, "buy")


def sell(p, q):
    return Trade("X", NOW, p, q, "sell")


class Book:
    def __init__(self, mid, bids=None, asks=None):
        self.mid, self.bids, self.asks = mid, dict(bids or {}), dict(asks or {})


# ------------------------------------------------------------ one second --
@pytest.mark.parametrize("before, after, trades, consumed, cancelled, replenished", [
    (10, 4, [buy(100.0, 6)], 6, 0, 0),         # traded away
    (10, 4, [buy(100.0, 2)], 2, 4, 0),         # mostly cancelled
    (10, 4, [buy(100.0, 8)], 6, 0, 2),         # executed more than removed: hidden refill (iceberg)
    (10, 10, [buy(100.0, 5)], 0, 0, 5),        # fully refilled
    (10, 15, [], 0, 0, 5),                     # new orders added
])
def test_second_record_splits_consumed_cancelled_replenished(before, after, trades, consumed, cancelled,
                                                              replenished):
    rec = second_record(1, 99.95, "resistance", L, {100.0: before}, {100.0: after}, trades, 1.0, R)
    assert rec["consumed"] == pytest.approx(consumed)
    assert rec["cancelled"] == pytest.approx(cancelled)
    assert rec["replenished"] == pytest.approx(replenished)


def test_trades_far_from_level_are_not_executed_into_it():
    rec = second_record(1, 99.0, "resistance", L, {100.0: 10}, {100.0: 10}, [buy(99.0, 5)], 1.0, R)
    assert rec["executed"] == 0 and rec["attack"] == 5


def test_support_uses_sells_as_attack():
    rec = second_record(1, 100.05, "support", L, {100.0: 10}, {100.0: 7}, [sell(100.0, 3), buy(100.1, 1)], 1.0, R)
    assert rec["consumed"] == 3 and rec["attack"] == 3 and rec["defence"] == 1


# --------------------------------------------------------- classification --
def run_seconds(kind, n, step):
    """step(i) -> (prev_depth, depth, trades, mid). Returns records."""
    recs, prev = [], None
    for i in range(n):
        p_depth, depth, trades, mid = step(i)
        rec = second_record(i, mid, kind, L, None if i == 0 else {100.0: p_depth}, {100.0: depth}, trades, 1.0, R)
        recs.append(rec)
    return recs


def judge(recs, kind="resistance"):
    t = recs[-1]["t"]
    c15 = classify(window(recs, t, 15, R), kind, L, R)[0]
    c60 = classify(window(recs, t, 60, R), kind, L, R)[0]
    return verdict(c15, c60)


def test_breakout_pressure_when_asks_are_consumed():
    recs = run_seconds("resistance", 60, lambda i: (200 - 3 * (i - 1), 200 - 3 * i, [buy(100.0, 3)],
                                                     99.95 + i * 0.0005))
    assert judge(recs) == ("breakout", "HIGH")


def test_reversal_pressure_when_asks_refill_and_price_stalls():
    recs = run_seconds("resistance", 60, lambda i: (200, 200, [buy(100.0, 3)], 99.95))
    assert judge(recs) == ("reversal", "HIGH")


def test_pulled_defence_is_inconclusive():
    recs = run_seconds("resistance", 60, lambda i: (200 - 3 * (i - 1), 200 - 3 * i, [buy(99.7, 1)], 99.95))
    w = window(recs, 59, 60, R)
    label, why = classify(w, "resistance", L, R)
    assert label == "inconclusive" and "pulled" in why


def test_no_clear_attacker_is_inconclusive():
    recs = run_seconds("resistance", 60, lambda i: (200, 200, [buy(100.0, 3), sell(99.9, 3)], 99.95))
    assert judge(recs) == ("inconclusive", "")


def test_support_breakdown_mirror():
    recs = run_seconds("support", 60, lambda i: (200 - 3 * (i - 1), 200 - 3 * i, [sell(100.0, 3)],
                                                  100.05 - i * 0.0005))
    assert judge(recs, "support") == ("breakout", "HIGH")


def test_not_enough_data():
    recs = run_seconds("resistance", 10, lambda i: (200, 200, [buy(100.0, 3)], 99.95))
    assert classify(window(recs, 9, 60, R), "resistance", L, R)[0] == "inconclusive"
    assert verdict("breakout", "inconclusive") == ("inconclusive", "")
    assert verdict("inconclusive", "reversal") == ("reversal", "MODERATE")


# --------------------------------------------------------------- analyzer --
def make(tmp_path, levels=(L,), auto=None):
    store = LevelStore(tmp_path / "my_levels.json")
    for p in levels:
        store.add("BTCUSDT.P", p)
    said = []
    a = SRAnalyzer(["BTCUSDT.P"], store, tmp_path, tmp_path / "raw", auto_levels=auto, out=said.append)
    return a, said


def test_visit_called_breakout_then_broke(tmp_path):
    a, said = make(tmp_path)
    t0 = 1_800_000_000
    for i in range(60):                                     # asks at 100 being eaten, price creeping up
        a.on_tick("BTCUSDT.P", t0 + i, Book(99.95 + i * 0.0005, asks={100.0: 200 - 3 * i}), [buy(100.0, 3)], 1.0)
    v = a.view("BTCUSDT.P")["watched"][0]
    assert v["verdict"] == "breakout" and v["strength"] == "HIGH" and v["visiting"]
    for i in range(60, 3700):                               # price goes through: +0.35%
        a.on_tick("BTCUSDT.P", t0 + i, Book(100.35, asks={100.5: 10}), [], 1.0)
    rows = pd.read_csv(tmp_path / "sr_visits.csv")
    assert len(rows) == 1 and rows["call"][0] == "breakout" and rows["outcome"][0] == "broke"
    assert pd.read_parquet(rows["raw_file"][0]).shape[0] > 3000
    sc = a.scorecard("BTCUSDT.P")
    assert sc["breakout_calls"] == 1 and sc["breakout_right"] == 1.0
    assert any("LONG - BREAKOUT (HIGH)" in s for s in said)
    # history is re-read after a restart
    a2, _ = make(tmp_path, levels=())
    assert a2.scorecard("BTCUSDT.P")["visits"] == 1


def test_visit_held(tmp_path):
    a, _ = make(tmp_path)
    t0 = 1_800_000_000
    for i in range(60):
        a.on_tick("BTCUSDT.P", t0 + i, Book(99.95, asks={100.0: 200}), [buy(100.0, 3)], 1.0)
    assert a.view("BTCUSDT.P")["watched"][0]["verdict"] == "reversal"
    for i in range(60, 3700):
        a.on_tick("BTCUSDT.P", t0 + i, Book(99.6, asks={100.0: 200}), [], 1.0)   # -0.4%: rejected
    assert a.scorecard("BTCUSDT.P")["reversal_right"] == 1.0


def test_far_levels_not_watched_and_auto_fallback(tmp_path):
    a, _ = make(tmp_path, levels=(), auto=lambda s: [L, 120.0])
    a.on_tick("BTCUSDT.P", 1, Book(99.5, asks={100.0: 10}), [], 1.0)
    v = a.view("BTCUSDT.P")
    assert v["origin"] == "auto" and [w["level"] for w in v["watched"]] == [L]
    assert not v["watched"][0]["visiting"]                     # 0.5% away: watched, not a visit yet


def test_blind_ticks_clear_history(tmp_path):
    a, _ = make(tmp_path)
    for i in range(30):
        a.on_tick("BTCUSDT.P", i, Book(99.95, asks={100.0: 200}), [buy(100.0, 3)], 1.0)
    a.on_tick("BTCUSDT.P", 30, None, [], 1.0)
    assert not a.watches["BTCUSDT.P"][L].records


def test_level_store(tmp_path):
    s = LevelStore(tmp_path / "x.json")
    s.add("BTCUSDT.P", 80000)
    s.add("BTCUSDT.P", 80001)          # within 0.01%: same level
    s.add("BTCUSDT.P", 82000)
    assert s.get("BTCUSDT.P") == [80000, 82000]
    s.remove("BTCUSDT.P", 80000)
    assert LevelStore(tmp_path / "x.json").get("BTCUSDT.P") == [82000]


def test_level_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import config
    monkeypatch.setenv("MM_LIVE_STREAM_ENABLED", "false")
    monkeypatch.setenv("MM_HISTORY_SYNC_ENABLED", "false")
    config.get_settings.cache_clear()
    from app.main import app
    with TestClient(app) as c:
        app.state.sr, _ = make(tmp_path, levels=())
        assert c.post("/api/levels/BTCUSDT.P", json={"price": 81500}).json()["levels"] == [81500]
        assert c.post("/api/levels/BTCUSDT.P", json={"price": -1}).status_code == 422
        assert c.get("/api/levels/BTCUSDT.P").json()["levels"] == [81500]
        assert c.delete("/api/levels/BTCUSDT.P?price=81500").json()["levels"] == []
        assert c.get("/api/levels/NOPE").status_code == 404
    config.get_settings.cache_clear()


def test_score_script_replays_visits(tmp_path):
    from scripts.score_sr_analyzer import load, score
    a, _ = make(tmp_path)
    t0 = 1_800_000_000
    for i in range(60):
        a.on_tick("BTCUSDT.P", t0 + i, Book(99.95 + i * 0.0005, asks={100.0: 200 - 3 * i}), [buy(100.0, 3)], 1.0)
    for i in range(60, 3700):
        a.on_tick("BTCUSDT.P", t0 + i, Book(100.35, asks={100.5: 10}), [], 1.0)
    visits = load(tmp_path / "sr_visits.csv")
    s = score(visits, AnalyzerRules())
    assert s["visits"] == 1 and s["breakout_calls"] == 1 and s["breakout_right"] == 1.0
    strict = score(visits, AnalyzerRules(attack_share=1.01))      # impossible threshold: no calls
    assert strict["breakout_calls"] == 0 and strict["inconclusive"] == 1


def test_quote_churn_is_not_counted_as_refill_or_pull():
    """Market maker moves 50 contracts between two prices every second while buyers eat 3/s at 100.0."""
    recs, prev = [], None
    for i in range(60):
        a, b = (50, 0) if i % 2 == 0 else (0, 50)
        z = {100.0: 200 - 3 * i, 100.05: a, 100.1: b}
        recs.append(second_record(i, 99.95 + i * 0.0005, "resistance", L, prev, z, [buy(100.0, 3)], 1.0, R))
        prev = z
    w = window(recs, 59, 60, R)
    assert w["replenished"] > 1000 and w["cancelled"] > 1000      # lots of churn ...
    assert judge(recs) == ("breakout", "HIGH")                    # ... but still read as consumption


# ------------------------------------------------------------ scored v2 --
from app.sranalyzer.scoring import (FailedCross, ImpactBaseline, ScoreRules, features, scores, survival,
                                    verdict2)

T0 = 1_800_000_000


def tb(t, p, q):
    return Trade("X", datetime.fromtimestamp(t, tz=timezone.utc), p, q, "buy")


def ts(t, p, q):
    return Trade("X", datetime.fromtimestamp(t, tz=timezone.utc), p, q, "sell")


def test_impact_baseline_and_survival():
    ib = ImpactBaseline()
    for m in range(12):                                  # each minute: $1M net buying moves price 2 bp
        ib.add(T0 + m * 60, 100.0 * (1 + 0.0002 * m), 10000.0, 0.0)
        ib.add(T0 + m * 60 + 30, 100.0 * (1 + 0.0002 * m), 0.0, 0.0)
    assert ib.value() == pytest.approx(2.0, rel=0.05)
    assert survival({100.0: 10, 100.1: 10}, {100.0: 10, 100.2: 50}) == pytest.approx(0.5)
    assert survival({}, {}) is None


def test_failed_cross_detector():
    sr, fc = ScoreRules(), FailedCross()
    assert not fc.update(0, 100.06, "resistance", L, 0.8, sr)     # through by 0.06%
    assert not fc.update(30, 99.97, "resistance", L, 0.7, sr)     # back below by 0.03%
    assert fc.update(40, 99.96, "resistance", L, 0.35, sr)        # sellers take over -> failed breakout
    fc2 = FailedCross()
    fc2.update(0, 100.06, "resistance", L, 0.8, sr)
    fc2.update(30, 99.97, "resistance", L, 0.7, sr)
    assert not fc2.update(200, 99.96, "resistance", L, 0.3, sr)   # flip came too late


def analyzer_run(a, n, step, start=T0):
    for i in range(n):
        mid, asks, trades = step(i)
        a.on_tick("BTCUSDT.P", start + i, Book(mid, asks=asks), trades, 1.0, {"resyncs": 0, "refreshes": 0})


def test_v2_strong_breakout(tmp_path):
    a, said = make(tmp_path)
    analyzer_run(a, 70, lambda i: (99.95 + i * 0.0006, {100.0: 200 - 2.5 * i, 100.05: 50},
                                   [tb(T0 + i, 100.0, 2.5)]))
    v2 = a.view("BTCUSDT.P")["watched"][0]["v2"]
    assert v2["verdict"] == "breakout" and v2["breakout_score"] >= 75 and v2["quality_ok"]
    assert any("[S/R v2]" in s and "LONG - BREAKOUT" in s for s in said)


def test_v2_strong_reversal_absorbed(tmp_path):
    a, _ = make(tmp_path)
    analyzer_run(a, 70, lambda i: (99.95, {100.0: 200, 100.05: 50}, [tb(T0 + i, 100.0, 3)]))
    v2 = a.view("BTCUSDT.P")["watched"][0]["v2"]
    assert v2["verdict"] == "reversal" and v2["reversal_score"] >= 75


def test_v2_failed_cross_gives_reversal(tmp_path):
    a, _ = make(tmp_path)

    def step(i):
        if i < 40:
            return 99.97, {100.0: 200}, [tb(T0 + i, 100.0, 3)]             # buyers attack
        if i < 45:
            return 100.08, {100.1: 200}, [tb(T0 + i, 100.08, 3)]           # poke through
        return 99.95, {100.0: 200}, [ts(T0 + i, 99.95, 4)]                 # back below, sellers take over
    analyzer_run(a, 75, step)
    w = a.watches["BTCUSDT.P"][L]
    assert w.cross.detected_at is not None and w.visit.failed_cross


def test_v2_quality_gate(tmp_path):
    a, _ = make(tmp_path)
    # normally on time, then trades suddenly stamped 10 s in the past: late feed
    analyzer_run(a, 70, lambda i: (99.95, {100.0: 200}, [tb(T0 + i - (10 if i >= 40 else 0), 100.0, 3)]))
    v2 = a.view("BTCUSDT.P")["watched"][0]["v2"]
    assert v2["verdict"] == "insufficient" and "later than usual" in v2["reason"]
    # order book rebuilt recently
    b, _ = make(tmp_path / "b")
    for i in range(70):
        b.on_tick("BTCUSDT.P", T0 + i, Book(99.95, asks={100.0: 200}), [tb(T0 + i, 100.0, 3)], 1.0,
                  {"resyncs": 1 if i > 30 else 0, "refreshes": 0})
    assert b.view("BTCUSDT.P")["watched"][0]["v2"]["verdict"] == "insufficient"


def test_pc_clock_offset_and_routine_refresh_do_not_block(tmp_path):
    """Seen live: 'trades arriving 8 s late' on every tick = this PC's clock is 8 s off MEXC's."""
    a, _ = make(tmp_path)
    analyzer_run(a, 70, lambda i: (99.95, {100.0: 200, 100.05: 50}, [tb(T0 + i - 8, 100.0, 3)]))
    v2 = a.view("BTCUSDT.P")["watched"][0]["v2"]
    assert v2["quality_ok"] and v2["verdict"] == "reversal"
    assert a.view("BTCUSDT.P")["clock_offset_s"] == pytest.approx(8)
    c, _ = make(tmp_path / "c")                     # routine 2-min refreshes are not gaps
    for i in range(70):
        c.on_tick("BTCUSDT.P", T0 + i, Book(99.95, asks={100.0: 200, 100.05: 50}), [tb(T0 + i, 100.0, 3)], 1.0,
                  {"resyncs": 0, "refreshes": i // 20})
    assert c.view("BTCUSDT.P")["watched"][0]["v2"]["quality_ok"]


def test_v2_conflicting_and_ablation():
    f = {"quality_ok": True, "distance_pct": 0.0, "hit_ratio": 1.0, "n_flow": 1.0, "n_consume": 1.0, "n_no_refill": 1.0, "n_depth_fall": 0.0,
         "n_refill": 0.0, "n_survive": 1.0, "n_progress": 0.0, "n_stall": 1.0, "n_impact_high": 0.0,
         "n_absorbed": 1.0, "persistence": 1.0, "failed_cross": False, "progress_pct": 0.0}
    sr = ScoreRules()
    b, r = scores(f, sr)
    assert b >= 60 and r >= 60 and verdict2(f, b, r, sr)[0] == "conflicting"
    b2, r2 = scores(f, ScoreRules(use_persistence=False))
    assert (b2, r2) != (b, r)
    assert verdict2({"quality_ok": False, "quality_reason": "x"}, None, None, sr) == ("insufficient", "x")


def test_overlapping_visits_flagged_and_excluded(tmp_path):
    a, _ = make(tmp_path, levels=(L, 100.08))
    analyzer_run(a, 3700, lambda i: (100.04, {100.1: 50}, [tb(T0 + i, 100.04, 0.1)]))
    rows = pd.read_csv(tmp_path / "sr_visits.csv")
    assert len(rows) == 2 and rows["overlap"].tolist().count(True) == 1
    sc = a.scorecard("BTCUSDT.P")
    assert sc["visits"] == 1 and sc["overlapping_excluded"] == 1


def test_score_script_compares_versions(tmp_path):
    from scripts.score_sr_analyzer import VARIANTS, load, score
    a, _ = make(tmp_path)
    analyzer_run(a, 70, lambda i: (99.95 + i * 0.0006, {100.0: 200 - 2.5 * i, 100.05: 50},
                                   [tb(T0 + i, 100.0, 2.5)]))
    analyzer_run(a, 3700, lambda i: (100.4, {100.5: 10}, [tb(T0 + 70 + i, 100.4, 0.1)]), start=T0 + 70)
    visits = load(tmp_path / "sr_visits.csv")
    assert len(visits) == 1 and visits[0]["outcome"] == "broke"
    for name, sr in VARIANTS.items():
        s = score(visits, AnalyzerRules(), sr)
        assert s["breakout_calls"] == 1 and s["breakout_right"] == 1.0, name


def test_v2_not_strong_when_price_has_left_the_level(tmp_path):
    """Seen on the test exchange: price 0.12% away and drifting off was called a STRONG reversal."""
    a, _ = make(tmp_path)
    analyzer_run(a, 70, lambda i: (99.88 - i * 0.0005, {100.0: 200, 100.05: 50}, [tb(T0 + i, 99.88, 3)]))
    v2 = a.view("BTCUSDT.P")["watched"][0]["v2"]
    assert v2["verdict"] in ("reversal_lean", "inconclusive") and "not being tested" in v2["reason"]


def test_v2_reversal_needs_attackers_hitting_the_level(tmp_path):
    a, _ = make(tmp_path)          # buyers active, but trading far below the level's asks
    analyzer_run(a, 70, lambda i: (99.95, {100.0: 200, 100.05: 50}, [tb(T0 + i, 99.7, 3)]))
    assert a.view("BTCUSDT.P")["watched"][0]["v2"]["verdict"] != "reversal"
