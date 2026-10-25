from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
import pytest

from app.adapters import mexc_futures
from app.adapters.mexc_futures import MexcFuturesAdapter
from app.ml.orb_filter import choose_threshold, make_model, oof_predictions, permutation_importance_auc
from app.strategy.orb import OrbConfig, run_orb
from app.strategy.orb_features import FEATURES, orb_features


def synthetic(days=60, step=15, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 1440 // step
    ts = pd.date_range("2026-03-02", periods=n, freq=f"{step}min", tz="UTC")
    close = 2400 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, close) * 1.001,
                         "low": np.minimum(o, close) * 0.999, "close": close,
                         "volume": rng.uniform(1, 3, n), "contracts": 1.0, "quote_volume": 1.0})


def test_features_do_not_use_data_after_breakout_candle():
    df = synthetic()
    trades = pd.DataFrame(run_orb(df, 15, OrbConfig(entry="close", stop="range", target="1x"))[0])
    assert len(trades) > 20
    funding = pd.DataFrame({"time": pd.date_range("2026-03-01", periods=200, freq="8h", tz="UTC"),
                            "rate": np.linspace(-1e-4, 1e-4, 200)})
    f1 = orb_features(df, 15, trades, "ETHUSDT.P", funding)
    k = 25
    known = trades["breakout_time"].iloc[k] + pd.Timedelta(minutes=15)
    altered = df.copy()
    future = altered["timestamp"] >= known
    altered.loc[future, ["open", "high", "low", "close"]] *= 1.5
    altered.loc[future, "volume"] *= 10
    fund2 = funding.copy()
    fund2.loc[fund2["time"] > known, "rate"] = 0.01
    f2 = orb_features(altered, 15, trades, "ETHUSDT.P", fund2)
    pd.testing.assert_series_equal(f1.iloc[k], f2.iloc[k])
    assert list(f1.columns) == FEATURES


def test_feature_signs_follow_trade_direction():
    df = synthetic(seed=3)
    trades = pd.DataFrame(run_orb(df, 15, OrbConfig(entry="close", stop="range", target="1x"))[0])
    f = orb_features(df, 15, trades, "BTCUSDT.P")
    longs = trades["side"] == "long"
    assert (f.loc[longs, "side_num"] == 1).all() and (f.loc[~longs, "side_num"] == -1).all()
    assert (f["bo_strength"] > 0).all()           # a breakout always closes beyond its edge
    assert f["is_eth"].eq(0).all() and f["funding_s"].isna().all()


def test_oof_predictions_never_use_later_rows():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    y = (X["a"] > 0).astype(int).to_numpy()
    p = oof_predictions(X, y, n_splits=5)
    assert np.isnan(p[:100]).all() and not np.isnan(p[100:]).any()   # first block has no history
    # If later labels change, earlier predictions must not
    y2 = y.copy()
    y2[400:] = 1 - y2[400:]
    p2 = oof_predictions(X, y2, n_splits=5)
    np.testing.assert_allclose(p[:400], p2[:400])


def test_threshold_choice_improves_and_keeps_minimum():
    p = np.linspace(0, 1, 200)
    r = np.where(p > 0.6, 1.0, -0.5)
    thr, info = choose_threshold(p, r, min_keep=50)
    assert info["kept"] >= 50 and info["mean_r"] > r.mean()
    assert thr >= 0.55


def test_model_handles_empty_feature_and_importance():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"signal": rng.normal(size=500), "empty": np.nan, "noise": rng.normal(size=500)})
    y = (X["signal"] + rng.normal(0, 0.5, 500) > 0).astype(int).to_numpy()
    m = make_model().fit(X, y)
    imp = dict(permutation_importance_auc(m, X, y, n_repeats=3))
    assert imp["signal"] > imp["noise"] and imp["empty"] == pytest.approx(0)


def test_funding_history_parsing_and_paging(monkeypatch):
    monkeypatch.setattr(mexc_futures, "MIN_SECONDS_BETWEEN_REQUESTS", 0)
    base = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def handler(req):
        page = int(req.url.params["page_num"])
        items = [{"symbol": "BTC_USDT", "fundingRate": 0.0001 * page,
                  "settleTime": base - ((page - 1) * 2 + i) * 8 * 3600 * 1000} for i in range(2)]
        return httpx.Response(200, json={"success": True, "code": 0, "data": {
            "pageSize": 2, "totalPage": 3, "currentPage": page, "resultList": items}})

    ad = MexcFuturesAdapter(client=httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler)))
    f = ad.fetch_funding_history("BTCUSDT.P", datetime(2026, 9, 1, tzinfo=timezone.utc), page_size=2)
    assert len(f) == 6 and f["time"].is_monotonic_increasing
    assert f["rate"].iloc[-1] == pytest.approx(0.0001)   # newest settlement, page 1
