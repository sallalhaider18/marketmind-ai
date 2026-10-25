import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.adapters.base import Trade
from app.data.live_stream import LiveMarketState, MexcFuturesStream
from app.orderbook.alerts import AlertEngine
from app.orderbook.book import BookSync, OrderBook
from app.orderbook.engine import OrderBookEngine, fmt_price
from app.orderbook.features import RollingMean, band_depth, book_metrics, imbalance, ladder, nice_step
from app.orderbook.large_orders import LargeOrderTracker
from app.orderbook.recorder import OrderBookRecorder


def snapshot_levels(mid=100.0, n=50, tick=0.1, qty=10.0):
    bids = [[round(mid - tick * (i + 1), 1), qty, 1] for i in range(n)]
    asks = [[round(mid + tick * (i + 1), 1), qty, 1] for i in range(n)]
    return bids, asks


# ------------------------------------------------------------------ book --
def test_book_apply_and_remove():
    b = OrderBook()
    b.load_snapshot([[99.9, 5, 1], [99.8, 3, 1]], [[100.1, 4, 1]], version=10)
    assert b.best_bid == 99.9 and b.best_ask == 100.1 and b.mid == pytest.approx(100.0)
    b.apply([[99.9, 0, 0], [99.95, 7, 2]], [[100.2, 1, 1]])
    assert 99.9 not in b.bids and b.best_bid == 99.95 and b.asks[100.2] == 1


def test_sync_buffers_then_replays_after_snapshot():
    s = BookSync("BTCUSDT.P")
    for v in (99, 100, 101, 102):
        s.on_update({"version": v, "bids": [[99.0, v, 1]], "asks": []})
    assert s.needs_snapshot and not s.synced
    bids, asks = snapshot_levels()
    assert s.on_snapshot(bids, asks, version=100)
    assert s.synced and s.book.version == 102 and s.book.bids[99.0] == 102  # 101, 102 replayed


def test_version_jumps_are_accepted():
    """MEXC bundles changes: versions rise in steps of several. That is not a loss."""
    s = BookSync("X")
    s.on_update({"version": 5, "bids": [], "asks": []})
    bids, asks = snapshot_levels()
    assert s.on_snapshot(bids, asks, version=5)
    s.on_update({"version": 9, "bids": [[99.85, 1, 1]], "asks": []})
    s.on_update({"version": 17, "bids": [[99.85, 4, 1]], "asks": []})
    assert s.synced and s.resyncs == 0 and s.book.version == 17 and s.book.bids[99.85] == 4


def test_snapshot_between_buffered_versions_joins():
    """Seen live: snapshot v390, buffered pushes ...388, 395, 401. Must sync."""
    s = BookSync("X")
    for v, q in ((380, 1), (388, 2), (395, 3), (401, 4)):
        s.on_update({"version": v, "bids": [[99.5, q, 1]], "asks": []})
    assert s.on_snapshot(*snapshot_levels(), version=390)
    assert s.book.version == 401 and s.book.bids[99.5] == 4


def test_refresh_measures_and_fixes_drift():
    s = BookSync("X")
    s.on_update({"version": 1, "bids": [], "asks": []})
    bids, asks = snapshot_levels()
    s.on_snapshot(bids, asks, version=1)
    s.on_update({"version": 4, "bids": [[99.9, 77, 1]], "asks": []})
    s.book.bids[99.8] = 999.0            # simulate silent corruption
    s.on_update({"version": 8, "bids": [[99.7, 55, 1]], "asks": []})
    # true book at v4: snapshot + the v4 change; refresh replays v8 on top
    true_bids = [[p, (77 if p == 99.9 else q), c] for p, q, c in bids]
    drift = s.refresh(true_bids, asks, version=4)
    assert drift == pytest.approx(1 / 40)   # one of 40 top levels was wrong
    assert s.book.bids[99.8] == 10 and s.book.bids[99.7] == 55 and s.book.version == 8
    assert s.refreshes == 1 and s.synced


def test_refresh_with_unusable_snapshot_changes_nothing():
    s = BookSync("X")
    s.on_update({"version": 10, "bids": [], "asks": []})
    s.on_snapshot(*snapshot_levels(), version=10)
    s.on_update({"version": 20, "bids": [[99.9, 3, 1]], "asks": []})
    before = dict(s.book.bids)
    assert s.refresh(*snapshot_levels(), version=5) is None   # older than what we kept
    assert s.book.bids == before


def test_sync_ignores_duplicates():
    s = BookSync("X")
    s.on_update({"version": 1, "bids": [], "asks": []})
    s.on_snapshot(*snapshot_levels(), version=1)
    s.on_update({"version": 1, "bids": [[1, 1, 1]], "asks": []})
    assert s.synced and 1.0 not in s.book.bids


def test_snapshot_older_than_buffer_is_retried():
    s = BookSync("X")
    s.on_update({"version": 50, "bids": [], "asks": []})
    assert not s.on_snapshot(*snapshot_levels(), version=40)     # 41..49 missing
    assert not s.synced and s.buffer


def test_crossed_book_triggers_resync():
    s = BookSync("X")
    s.on_update({"version": 1, "bids": [], "asks": []})
    s.on_snapshot(*snapshot_levels(), version=1)
    s.on_update({"version": 2, "bids": [[100.5, 3, 1]], "asks": []})   # bid above best ask
    assert not s.synced and "crossed" in s.last_resync_reason


# -------------------------------------------------------------- features --
def test_band_depth_and_imbalance():
    b = OrderBook()
    bids, asks = snapshot_levels(mid=100.0, n=50, tick=0.1, qty=10)
    asks = [[p, 5, 1] for p, _, _ in asks]
    b.load_snapshot(bids, asks, 1)
    bid, ask = band_depth(b, 0.01)                    # within ±1.0
    assert bid == pytest.approx(100) and ask == pytest.approx(50)
    assert imbalance(bid, ask) == pytest.approx(1 / 3)
    m = book_metrics(b, contract_size=0.5)
    assert m["spread"] == pytest.approx(0.2) and m["bands"][-1]["bid"] == pytest.approx(50)
    assert all(x["complete"] for x in m["bands"])     # snapshot covers ±5%


def test_metrics_flag_incomplete_bands():
    b = OrderBook()
    b.load_snapshot(*snapshot_levels(n=3), 1)         # only ±0.3% known
    flags = [x["complete"] for x in book_metrics(b, 1)["bands"]]
    assert flags == [True, True, False, False]


@pytest.mark.parametrize("raw,step", [(8.18, 10), (0.24, 0.25), (1.7, 2), (0.031, 0.05)])
def test_nice_step(raw, step):
    assert nice_step(raw) == pytest.approx(step)


def test_ladder_groups_levels():
    b = OrderBook()
    b.load_snapshot(*snapshot_levels(mid=100, n=50, tick=0.1, qty=1), 1)
    lad = ladder(b, 1.0, rows=3, step=0.5)
    assert lad["bids"][0]["price"] == pytest.approx(99.5)
    assert lad["asks"][0]["price"] == pytest.approx(100.5)
    assert sum(r["size"] for r in lad["bids"]) == pytest.approx(15)  # 99.9..98.5 (3 buckets x 5 levels)


def test_rolling_mean_windows():
    r = RollingMean()
    for t in range(60):
        r.add(t, 1.0 if t >= 50 else -1.0)
    assert r.mean(5, 59) == 1.0 and r.mean(60, 59) == pytest.approx(-40 / 60)
    assert r.coverage(60, 59) == pytest.approx(59 / 60)


# ---------------------------------------------------------- large orders --
def book_with_wall(price=99.7, size=500, side="bid"):
    b = OrderBook()
    bids, asks = snapshot_levels(qty=10)
    target = bids if side == "bid" else asks
    for lv in target:
        if lv[0] == price:
            lv[1] = size
    b.load_snapshot(bids, asks, 1)
    return b


def trade(price, side, qty):
    return Trade("X", datetime.now(timezone.utc), price, qty, side)


def test_large_order_detected_and_persisting():
    tr = LargeOrderTracker()
    b = book_with_wall()
    for t in range(0, 31):
        tr.update(float(t), b, [])
    assert list(tr.active) == [("bid", 99.7)]
    o = tr.active[("bid", 99.7)]
    assert o.status(30) == "holding" and o.age(30) == 30
    v = tr.view(30, b.mid, 0.0001)
    assert v["active"][0]["size"] == pytest.approx(0.05) and v["active"][0]["times_typical"] == pytest.approx(50)


def test_large_order_pulled_vs_filled():
    tr = LargeOrderTracker(grace_seconds=2)
    b = book_with_wall(99.7)
    for t in range(15):
        tr.update(float(t), b, [])
    b.bids.pop(99.7)                                   # disappears, price far away
    tr.update(15.0, b, [])
    ended = tr.update(17.0, b, [])
    assert ended and ended[0].end_reason == "pulled"

    tr = LargeOrderTracker(grace_seconds=2)
    b = book_with_wall(99.7)
    for t in range(15):
        tr.update(float(t), b, [])
    tr.update(15.0, b, [trade(99.7, "sell", 300)])     # sellers hit it
    b.bids.pop(99.7)
    tr.update(16.0, b, [trade(99.7, "sell", 200)])
    ended = tr.update(18.0, b, [])
    assert ended[0].end_reason == "filled" and ended[0].hit == 500


def test_absorption_status():
    tr = LargeOrderTracker()
    b = book_with_wall(99.7, 500)
    for t in range(12):
        tr.update(float(t), b, [trade(99.7, "sell", 15)])
    assert tr.active[("bid", 99.7)].status(12) == "absorbing"


def test_brief_flicker_does_not_end_order():
    tr = LargeOrderTracker(grace_seconds=2)
    b = book_with_wall()
    tr.update(0, b, [])
    size = b.bids.pop(99.7)
    tr.update(1, b, [])
    b.bids[99.7] = size
    tr.update(2, b, [])
    assert ("bid", 99.7) in tr.active and not tr.ended


# ---------------------------------------------------------------- alerts --
def test_alerts_hold_once_and_cooldown():
    tr = LargeOrderTracker()
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    b = book_with_wall(99.5)
    for t in range(0, 60):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [], None, None, 0, None)
    holds = [x for x in a.recent if x["kind"] == "large_order"]
    assert len(holds) == 1 and "held 30 s" in holds[0]["text"]


def test_removal_alert_only_after_hold_alert():
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    for hold_for, expect in ((15, 0), (40, 1)):
        tr = LargeOrderTracker(grace_seconds=2)
        b = book_with_wall(99.8)
        a.recent.clear()
        for t in range(hold_for):
            tr.update(float(t), b, [])
            a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [], None, None, 0, None)
        b.bids.pop(99.8)
        for t in (hold_for, hold_for + 2):
            ended = tr.update(float(t), b, [])
            a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), ended, None, None, 0, None)
        assert len([x for x in a.recent if x["kind"] == "pulled"]) == expect


def test_tracker_band_capped_by_snapshot_coverage():
    tr = LargeOrderTracker(band_pct=0.005)
    b = OrderBook()
    bids, asks = snapshot_levels(n=3, qty=10)      # covers only ±0.3%
    b.load_snapshot(bids, asks, 1)
    for p in (99.5, 99.4):                          # levels learned later, outside coverage
        b.bids[p] = 5000
    tr._candidates(b)
    assert tr.band_used == pytest.approx(b.coverage_pct())


def test_only_the_strongest_order_per_side_is_alerted():
    """Several large bids: one alert, for the biggest. Smaller 'large' orders stay silent."""
    tr = LargeOrderTracker()
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=500)
    b = OrderBook()
    bids, asks = snapshot_levels(qty=10)
    for lv, q in zip(bids[2:6], (120, 900, 150, 130)):
        lv[1] = q
    asks[3][1] = 100                        # large ask, but only 100 coins (< 500 minimum)
    b.load_snapshot(bids, asks, 1)
    for t in range(40):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [], threshold=tr.threshold)
    holds = [x for x in a.recent if x["kind"] == "large_order"]
    assert len(holds) == 1
    assert holds[0]["text"].startswith("Strongest bid: 900.00 BTC ($90k) at 99.60")


def test_strongest_alert_repeats_only_when_clearly_bigger():
    tr = LargeOrderTracker()
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    b = book_with_wall(99.7, 500)
    for t in range(40):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [], threshold=tr.threshold)
    b.bids[99.7] = 0; b.bids.pop(99.7)
    b.bids[99.6] = 600                     # a bit bigger, soon after: no new alert
    for t in range(40, 90):
        ended = tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), ended, threshold=tr.threshold)
    assert len([x for x in a.recent if x["kind"] == "large_order"]) == 1
    b.bids[99.5] = 1200                    # clearly bigger: reported
    for t in range(90, 130):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [], threshold=tr.threshold)
    holds = [x for x in a.recent if x["kind"] == "large_order"]
    assert len(holds) == 2 and "1,200.00 BTC ($120k) at 99.50" in holds[0]["text"]


def test_old_small_order_is_not_reported_while_biggest_is_young():
    """Seen live: a 0.02 BTC leftover was alerted as 'strongest' because the real big ones were new."""
    tr = LargeOrderTracker()
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    b = book_with_wall(99.7, 200)
    for t in range(20):
        tr.update(float(t), b, [])
    b.bids[99.5] = 2000                    # new, much bigger
    for t in range(20, 45):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [])
    holds = [x for x in a.recent if x["kind"] == "large_order"]
    assert len(holds) == 0                 # 99.7 is old but not the biggest; 99.5 is not 30 s old yet
    for t in range(45, 52):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [])
    holds = [x for x in a.recent if x["kind"] == "large_order"]
    assert len(holds) == 1 and "at 99.50" in holds[0]["text"]


def test_order_shrinking_to_a_sliver_ends():
    tr = LargeOrderTracker(grace_seconds=2)
    b = book_with_wall(99.7, 1000)
    tr.update(0, b, [])
    b.bids[99.7] = 100                     # 10% of peak, still above the large size
    for t in (1, 4):
        tr.update(float(t), b, [])
    assert tr.ended and tr.ended[0].price == 99.7          # the big order ended
    assert tr.active[("bid", 99.7)].first_seen == 4.0         # what is left is tracked as a new, small one


def test_no_book_pressure_alerts():
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    a.evaluate(1, 100, 1, [], [], 0.4, 0.35, 1.0, 0.9)
    assert not a.recent


# -------------------------------------------------------------- recorder --
def test_recorder_writes_parquet(tmp_path):
    rec = OrderBookRecorder(tmp_path, levels=5, flush_seconds=10)
    b = OrderBook()
    b.load_snapshot(*snapshot_levels(), 7)
    m = book_metrics(b, 0.0001)
    for i in range(12):
        rec.add("BTCUSDT.P", 1_700_000_000 + i, b, m, 0.0001)
    assert rec.due(1_700_000_011) == ["BTCUSDT.P"]
    path = rec.flush("BTCUSDT.P")
    df = pd.read_parquet(path)
    assert len(df) == 12 and len(df["bid_px"].iloc[0]) == 5 and "imbalance_25bp" in df.columns
    assert df["version"].iloc[0] == 7


# --------------------------------------------- stream + engine integration --
def depth_msg(v, bids=(), asks=()):
    return json.dumps({"channel": "push.depth", "symbol": "BTC_USDT",
                       "data": {"version": v, "bids": list(bids), "asks": list(asks)}})


def test_stream_routes_depth_and_engine_builds_view(tmp_path):
    state = LiveMarketState(["BTCUSDT.P"], stale_after=10)
    state.connected = True
    stream = MexcFuturesStream(state)
    assert {"method": "sub.depth", "param": {"symbol": "BTC_USDT"}} in stream.subscription_messages()
    stream.handle_message(depth_msg(10))
    book = state.symbols["BTCUSDT.P"].book
    assert book.needs_snapshot
    bids, asks = snapshot_levels(mid=81000, n=200, tick=0.1, qty=100)
    book.on_snapshot(bids, asks, version=10)
    stream.handle_message(depth_msg(11, bids=[[80999.9, 9000, 3]]))
    assert book.synced and book.book.bids[80999.9] == 9000

    clock = [0.0]
    engine = OrderBookEngine(state, contract_size=lambda s: 0.0001,
                             recorder=OrderBookRecorder(tmp_path), clock=lambda: clock[0])
    view = engine.step("BTCUSDT.P")
    assert view["status"] == "live"
    assert view["metrics"]["best_bid"] == pytest.approx(80999.9)
    assert view["large_orders"]["active"][0]["price"] == pytest.approx(80999.9)
    assert view["ladder"]["step"] == 10
    assert engine.recorder.pending("BTCUSDT.P") == 1


def test_engine_reports_syncing_when_not_synced():
    state = LiveMarketState(["BTCUSDT.P"])
    state.connected = True
    engine = OrderBookEngine(state, contract_size=lambda s: 0.0001)
    assert engine.step("BTCUSDT.P")["status"] == "syncing"


def test_min_wall_defaults_per_coin():
    assert AlertEngine("BTCUSDT.P", "BTC", fmt_price).min_wall == 100
    assert AlertEngine("ETHUSDT.P", "ETH", fmt_price).min_wall == 1000


def _reported_wall(side, price, size=500):
    tr = LargeOrderTracker(grace_seconds=2)
    a = AlertEngine("BTCUSDT.P", "BTC", fmt_price, min_wall=0)
    b = book_with_wall(price, size, side)
    for t in range(35):
        tr.update(float(t), b, [])
        a.evaluate(float(t), b.mid, 1.0, list(tr.active.values()), [])
    assert any(x["kind"] == "large_order" for x in a.recent)
    return tr, a, b


def _eat_ask(tr, a, b, price, traded, then_mid, t0=35, until=40):
    """Buyers trade `traded` into the ask at `price`, it disappears, then mid sits at `then_mid`."""
    b.asks.pop(price)
    trades = [Trade("BTCUSDT.P", datetime.now(timezone.utc), price, traded, "buy")]
    for t in range(t0, until):
        ended = tr.update(float(t), b, trades if t == t0 else [])
        a.evaluate(float(t), then_mid if t > t0 else b.mid, 1.0, list(tr.active.values()), ended)


def test_wall_eaten_and_broken_through():
    tr, a, b = _reported_wall("ask", 100.3)
    _eat_ask(tr, a, b, 100.3, traded=400, then_mid=100.4)
    br = [x for x in a.recent if x["kind"] == "breakout"]
    assert len(br) == 1 and br[0]["tone"] == "up" and br[0]["text"].startswith("BROKE THROUGH: strongest ask")


def test_wall_eaten_without_break_reports_after_window():
    tr, a, b = _reported_wall("ask", 100.3)
    _eat_ask(tr, a, b, 100.3, traded=400, then_mid=100.0, until=40)
    assert not [x for x in a.recent if x["kind"] in ("breakout", "filled")]
    for t in range(40, 170, 10):
        a.evaluate(float(t), 100.0, 1.0, list(tr.active.values()), [])
    f = [x for x in a.recent if x["kind"] == "filled"]
    assert len(f) == 1 and "did not break through" in f[0]["text"]
    assert not [x for x in a.recent if x["kind"] == "breakout"]


def test_wall_gone_with_little_traded_is_called_pulled():
    tr, a, b = _reported_wall("ask", 100.3)
    _eat_ask(tr, a, b, 100.3, traded=20, then_mid=100.4)
    assert not [x for x in a.recent if x["kind"] == "breakout"]
    assert any("pulled at the last moment" in x["text"] for x in a.recent)
