from datetime import datetime, timezone

import pytest

from app.adapters.base import Trade
from app.orderflow.flow import OrderFlow

NOW = (datetime.now(timezone.utc).timestamp() // 60) * 60 + 30     # middle of the current minute


def tr(sec_ago, price, coins, side):
    ts = datetime.fromtimestamp(NOW - sec_ago, tz=timezone.utc)
    return Trade("BTCUSDT.P", ts, price, coins / 0.0001, side)


def feed(f, trades):
    f.add(trades, 0.0001)


def minutes_of(start_price, end_price, buy, sell, minutes=20):
    """One buy and one sell per minute, price moving linearly."""
    out = []
    for i in range(minutes):
        p = start_price + (end_price - start_price) * i / (minutes - 1)
        ago = (minutes - 1 - i) * 60 + 5
        out += [tr(ago + 0.5, p, buy, "buy"), tr(ago, p, sell, "sell")]
    return out


def test_fills_of_one_market_order_are_merged_into_one_big_order():
    f = OrderFlow("BTC")
    ts = datetime.fromtimestamp(NOW, tz=timezone.utc)
    feed(f, [Trade("BTCUSDT.P", ts, 100.0, 40000, "buy"), Trade("BTCUSDT.P", ts, 100.1, 40000, "buy"),
             Trade("BTCUSDT.P", ts, 100.2, 40000, "buy")])     # 3 fills x 4 BTC, same ms
    feed(f, [tr(-0.5, 100.2, 1, "sell")])                       # small, different order
    v = f.view(now=NOW + 5)
    assert len(v["big_orders"]) == 1
    big = v["big_orders"][0]
    assert big["side"] == "buy" and big["size"] == pytest.approx(12) and big["to"] == 100.2


def test_cvd_adds_buys_and_subtracts_sells():
    f = OrderFlow("BTC")
    feed(f, minutes_of(100, 100, buy=5, sell=2, minutes=3))
    cvd = f.view(now=NOW)["cvd"]
    assert [round(c["cvd"]) for c in cvd] == [3, 6, 9]


def test_buyers_in_control_when_both_windows_agree():
    f = OrderFlow("BTC")
    feed(f, minutes_of(80000, 80400, buy=30, sell=10))
    v = f.view(now=NOW)
    assert v["overall"]["label"] == "Buyers in control" and v["overall"]["summary"].startswith("UP")


def test_weak_rise_warns():
    f = OrderFlow("BTC")
    feed(f, minutes_of(80000, 80400, buy=8, sell=30))
    assert f.view(now=NOW)["windows"][0]["reading"]["label"] == "Weak rise"


def test_sellers_absorbed_is_bullish():
    f = OrderFlow("BTC")
    feed(f, minutes_of(80000, 80010, buy=5, sell=40))
    r = f.view(now=NOW)["windows"][1]["reading"]
    assert r["label"] == "Sellers absorbed" and r["tone"] == "up"


def test_quiet_market_gives_no_reading():
    f = OrderFlow("BTC")
    feed(f, minutes_of(80000, 80400, buy=0.3, sell=0.1))
    assert f.view(now=NOW)["overall"]["summary"] == "NO CLEAR SIDE"
