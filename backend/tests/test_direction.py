import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import Costs
from app.direction.features import FEATURES, SEQ_COLS, build_features, sequences
from app.direction.labels import label_decisions
from app.direction.live import PaperTrader, Predictor
from app.direction.models import TORCH, TreeModel
from app.direction.pipeline import select_trades

NO_COSTS = Costs(0, 0, 0, 0)


def walk(days=40, seed=0, base=80000.0, start="2026-06-01"):
    rng = np.random.default_rng(seed)
    n = days * 48
    ts = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    close = base * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, close) * 1.002,
                         "low": np.minimum(o, close) * 0.998, "close": close,
                         "volume": rng.uniform(1, 3, n), "contracts": 1.0, "quote_volume": 1.0})


# ------------------------------------------------------------------ features --
def test_features_use_only_closed_candles():
    df, other = walk(seed=1), walk(seed=2, base=2400)
    f1 = build_features(df, "BTCUSDT.P", other)
    k = int(np.flatnonzero(f1["is_decision"].to_numpy())[200])
    known = f1["decision_time"].iloc[k]
    alt, alt_o = df.copy(), other.copy()
    for d in (alt, alt_o):
        fut = d["timestamp"] >= known
        d.loc[fut, ["open", "high", "low", "close"]] *= 1.4
        d.loc[fut, "volume"] *= 9
    f2 = build_features(alt, "BTCUSDT.P", alt_o)
    pd.testing.assert_series_equal(f1[FEATURES + SEQ_COLS].iloc[k], f2[FEATURES + SEQ_COLS].iloc[k])


def test_decision_candles_are_new_york_session_only():
    f = build_features(walk(), "BTCUSDT.P")
    t = pd.DatetimeIndex(f["decision_time"][f["is_decision"]]).tz_convert("America/New_York")
    mins = t.hour * 60 + t.minute
    assert mins.min() == 10 * 60 and mins.max() == 16 * 60        # closes 10:00 .. 16:00 NY
    assert len(set(t.weekday)) == 7                                # every day of the week


def test_sequences_shape_and_order():
    f = build_features(walk(), "BTCUSDT.P")
    s = sequences(f, np.array([100, 500]), seq_len=48)
    assert s.shape == (2, 48, len(SEQ_COLS))
    assert s[1, -1, 0] == pytest.approx(f["s_ret"].iloc[500], abs=1e-5)   # last step = decision candle


# -------------------------------------------------------------------- labels --
def path(bars):
    ts = pd.date_range("2026-07-15 14:00", periods=len(bars), freq="30min", tz="UTC")   # 10:00 NY (EDT)
    return pd.DataFrame({"timestamp": ts, "open": [b[0] for b in bars], "high": [b[1] for b in bars],
                         "low": [b[2] for b in bars], "close": [b[3] for b in bars], "volume": 1.0})


def label(bars, dist=1000):
    df = path(bars)
    return label_decisions(df, np.array([0]), pd.Series(["2026-07-15"]), dist, NO_COSTS).iloc[0]


def test_label_up_first():
    r = label([(80000, 80100, 79900, 80000), (80000, 80500, 79700, 80400), (80400, 81100, 80300, 81000)]
              + [(81000, 81050, 80950, 81000)] * 5)
    assert r["up_first"] == 1 and r["long_r"] == pytest.approx(1) and r["short_r"] == pytest.approx(-1)
    assert r["long_exit"] == "target" and r["short_exit"] == "stop"


def test_label_both_in_one_candle_is_unknown_and_costs_a_loss_both_ways():
    r = label([(80000, 80100, 79900, 80000), (80000, 81200, 78800, 80000)] + [(80000, 80050, 79950, 80000)] * 3)
    assert np.isnan(r["up_first"]) and r["long_r"] == pytest.approx(-1) and r["short_r"] == pytest.approx(-1)


def test_label_time_exit_next_day_929():
    bars = [(80000, 80100, 79900, 80000)] + [(80000, 80300, 79800, 80200)] * 60
    df = path(bars)
    lab = label_decisions(df, np.array([0]), pd.Series(["2026-07-15"]), 1000, NO_COSTS).iloc[0]
    end_ny = df["timestamp"].iloc[int(lab["horizon_end"])].tz_convert("America/New_York")
    assert (end_ny.day, end_ny.hour, end_ny.minute) == (16, 9, 0)          # candle 9:00-9:30 next day
    assert lab["long_exit"] == "next-day close (9:29 ET)" and lab["long_r"] == pytest.approx(0.2)
    assert np.isnan(lab["up_first"])


def test_label_costs_reduce_result():
    bars = [(80000, 80100, 79900, 80000), (80000, 81100, 79900, 81000)] + [(81000,) * 4] * 3
    df = path(bars)
    lab = label_decisions(df, np.array([0]), pd.Series(["2026-07-15"]), 1000, Costs()).iloc[0]
    assert 0.9 < lab["long_r"] < 1.0


# ------------------------------------------------------------- daily rule ---
def test_first_confident_signal_per_coin_per_day():
    data = pd.DataFrame({
        "decision_time": pd.date_range("2026-07-15 14:00", periods=6, freq="30min", tz="UTC").tolist() * 2,
        "symbol": ["BTCUSDT.P"] * 6 + ["ETHUSDT.P"] * 6, "ny_date": "2026-07-15",
        "long_r": 1.0, "short_r": -1.0, "long_exit": "target", "short_exit": "stop", "price": 1.0})
    p = np.array([0.55, 0.70, 0.90, 0.20, 0.5, 0.5,   0.52, 0.48, 0.30, 0.95, 0.5, 0.5])
    tr = select_trades(data, p, thr=0.15)
    assert len(tr) == 2
    btc = tr[tr["symbol"] == "BTCUSDT.P"].iloc[0]
    eth = tr[tr["symbol"] == "ETHUSDT.P"].iloc[0]
    assert btc["side"] == "long" and btc["p_up"] == 0.70          # first >= 65%, not the day's highest
    assert eth["side"] == "short" and eth["r"] == -1.0


# ----------------------------------------------------------------- models ---
def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    seq = rng.normal(size=(n, 48, len(SEQ_COLS))).astype(np.float32)
    signal = X["r4h_1"] + seq[:, -6:, 0].sum(axis=1) * 0.5
    y = (signal + rng.normal(0, 1, n) > 0).astype(int)
    return X, seq, y


def test_tree_model_learns_signal():
    X, S, y = _toy()
    m = TreeModel().fit(X[:2000], S[:2000], y[:2000])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[2000:], m.predict(X[2000:], S[2000:])) > 0.65


@pytest.mark.skipif(not TORCH, reason="PyTorch not installed")
def test_lstm_learns_sequence_signal_and_roundtrips():
    from sklearn.metrics import roc_auc_score
    from app.direction.models import LSTMModel
    X, S, y = _toy()
    m = LSTMModel(epochs=8).fit(X[:2000], S[:2000], y[:2000])
    p = m.predict(X[2000:], S[2000:])
    assert roc_auc_score(y[2000:], p) > 0.65
    m2 = LSTMModel.from_bytes(m.to_bytes())
    np.testing.assert_allclose(m2.predict(X[2000:], S[2000:]), p, rtol=1e-5)


# ------------------------------------------------------------ live trader ---
class FakeMarket:
    def __init__(self):
        self.df = {"BTCUSDT.P": walk(days=60, seed=5, start="2026-05-20"),
                   "ETHUSDT.P": walk(days=60, seed=6, base=2400, start="2026-05-20")}
        self.price = {"BTCUSDT.P": 80000.0, "ETHUSDT.P": 2400.0}
        self.range = {}

    def candles_30m(self):
        return self.df

    def funding(self, symbol):
        return None

    def recent_range(self, symbol, since):
        return self.range.get(symbol, (self.price[symbol],) * 3)

    def last_price(self, symbol):
        return self.price[symbol]


def stub(p_btc, p_eth=0.5):
    def predict(X, seq):
        p = p_btc if X["is_eth"].iloc[0] == 0 else p_eth
        return {"ml": p, "dl": p, "ensemble": p}
    return predict


def make_trader(tmp_path, predictor, lines):
    return PaperTrader(FakeMarket(), predictor, "ensemble", 0.1, {"BTCUSDT.P": 1000, "ETHUSDT.P": 100},
                       tmp_path / "state.json", tmp_path / "trades.csv", out=lines.append)


def at(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC").to_pydatetime()


def test_live_trade_open_target_and_one_per_day(tmp_path):
    lines = []
    tr = make_trader(tmp_path, stub(0.72), lines)
    m = tr.m
    m.df = {k: v[v["timestamp"] < pd.Timestamp("2026-07-15 14:30", tz="UTC")] for k, v in m.df.items()}
    tr.step(at("2026-07-15 10:00:20"))                 # 9:30 candle closed -> decision
    assert "BTCUSDT.P" in tr.state.open and any(">>> PAPER LONG" in x for x in lines)
    assert any("no trade" in x and "ETHUSDT.P" in x for x in lines)
    t = tr.state.open["BTCUSDT.P"]
    assert t["stop"] == 79000 and t["target"] == 81000

    m.range["BTCUSDT.P"] = (81050.0, 80500.0, 81000.0)  # target reached
    m.df = {k: v[v["timestamp"] < pd.Timestamp("2026-07-15 15:00", tz="UTC")] for k, v in FakeMarket().df.items()}
    tr.step(at("2026-07-15 10:30:20"))
    assert "BTCUSDT.P" not in tr.state.open and tr.state.closed == 1 and tr.state.wins == 1
    assert any("<<< CLOSED LONG" in x and "WIN" in x for x in lines)
    assert any("already traded today" in x for x in lines)            # same day: no second trade
    log = pd.read_csv(tmp_path / "trades.csv")
    assert log["r"].iloc[0] == pytest.approx(1.0, abs=0.02)

    # state survives a restart
    tr2 = make_trader(tmp_path, stub(0.72), [])
    assert tr2.state.closed == 1 and tr2.state.traded_on["BTCUSDT.P"] == "2026-07-15"


def test_live_short_stop_and_time_exit(tmp_path):
    lines = []
    tr = make_trader(tmp_path, stub(0.5, p_eth=0.2), lines)
    tr.m.df = {k: v[v["timestamp"] < pd.Timestamp("2026-07-15 14:30", tz="UTC")] for k, v in tr.m.df.items()}
    tr.step(at("2026-07-15 10:00:20"))
    assert tr.state.open["ETHUSDT.P"]["side"] == "short"
    tr.m.price["ETHUSDT.P"] = 2390.0
    tr.step(at("2026-07-16 09:29:30"))                # next day 9:29 NY: time exit
    assert tr.state.closed == 1 and any("next-day close" in x for x in lines)


def test_no_decisions_outside_new_york_session(tmp_path):
    lines = []
    tr = make_trader(tmp_path, stub(0.9), lines)
    tr.step(at("2026-07-15 08:00:20"))
    tr.step(at("2026-07-15 17:00:20"))
    assert lines == [] and not tr.state.open
    assert PaperTrader.next_decision(at("2026-07-15 17:00:20")) == pd.Timestamp(at("2026-07-16 10:00"))


def test_predictor_loads_saved_models(tmp_path):
    X, S, y = _toy(800)
    joblib.dump(TreeModel().fit(X, S, y), tmp_path / "ml.joblib")
    meta = {"models": ["ml"], "thresholds": {"ml": 0.12}, "distance": {"BTCUSDT.P": 1000},
            "live_model": "ml", "proven": False, "trained_until": "2026-10-01"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    pred = Predictor(tmp_path)
    out = pred(X.iloc[[0]], S[:1])
    assert pred.model == "ml" and pred.threshold == 0.12 and 0 < out["ml"] < 1
