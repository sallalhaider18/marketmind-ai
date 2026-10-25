import asyncio
import json
import time

import pytest
import websockets

from app.data.live_stream import LiveMarketState, MexcFuturesStream, parse_deal_push, parse_kline_push


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def now_ms(): return int(time.time() * 1000)


KLINE = {"channel": "push.kline", "symbol": "BTC_USDT",
         "data": {"symbol": "BTC_USDT", "interval": "Min1", "t": 1700000040,
                  "o": 100, "h": 102, "l": 99, "c": 101, "q": 500, "a": 50.5}}


def deal(T, v, p=101.0):
    return {"channel": "push.deal", "symbol": "BTC_USDT", "data": {"p": p, "v": v, "T": T, "O": 1, "M": 1, "t": now_ms()}}


def test_parse_kline_push():
    c = parse_kline_push(KLINE)
    assert c["close"] == 101 and c["contracts"] == 500 and c["timestamp"].timestamp() == 1700000040


def test_parse_deal_push_single_and_list():
    assert parse_deal_push(deal(1, 5), "BTCUSDT.P")[0].side == "buy"
    msg = {"channel": "push.deal", "symbol": "BTC_USDT", "data": [deal(1, 2)["data"], deal(2, 3)["data"]]}
    trades = parse_deal_push(msg, "BTCUSDT.P")
    assert [t.side for t in trades] == ["buy", "sell"]


def test_parse_deal_skips_malformed():
    assert parse_deal_push({"data": {"p": 1}}, "X") == []


def test_handle_messages_update_state_and_flow():
    clock = FakeClock()
    st = LiveMarketState(["BTCUSDT.P", "ETHUSDT.P"], stale_after=10, clock=clock)
    st.connected = True
    s = MexcFuturesStream(st)
    s.handle_message(json.dumps(KLINE))
    s.handle_message(json.dumps(deal(1, 30)))
    s.handle_message(json.dumps(deal(2, 10)))
    s.handle_message(json.dumps({"channel": "pong", "data": 1}))
    s.handle_message("not json")
    s.handle_message(json.dumps({"channel": "push.deal", "symbol": "DOGE_USDT", "data": deal(1, 1)["data"]}))

    snap = st.snapshot("BTCUSDT.P")
    assert snap["live_candle"]["close"] == 101
    assert snap["last_price"] == 101.0
    f = snap["trade_flow_60s"]
    assert f["buy_contracts"] == 30 and f["sell_contracts"] == 10 and f["delta_contracts"] == 20
    assert f["buy_ratio"] == pytest.approx(0.75)
    assert not snap["stale"]
    assert st.is_stale("ETHUSDT.P")  # never received data


def test_stale_after_silence_and_on_disconnect():
    clock = FakeClock()
    st = LiveMarketState(["BTCUSDT.P"], stale_after=10, clock=clock)
    st.connected = True
    MexcFuturesStream(st).handle_message(json.dumps(deal(1, 1)))
    assert not st.is_stale("BTCUSDT.P")
    clock.t += 11
    assert st.is_stale("BTCUSDT.P")
    clock.t -= 11
    st.connected = False
    assert st.is_stale("BTCUSDT.P")


def test_subscriptions():
    msgs = MexcFuturesStream(LiveMarketState(["BTCUSDT.P", "ETHUSDT.P"])).subscription_messages()
    assert {"method": "sub.kline", "param": {"symbol": "ETH_USDT", "interval": "Min1"}} in msgs
    assert {"method": "sub.deal", "param": {"symbol": "BTC_USDT"}} in msgs


@pytest.mark.asyncio
async def test_end_to_end_with_local_server_and_reconnect():
    """A local WebSocket server plays MEXC: it checks subscriptions, sends data,
    then drops the connection. The client must reconnect and keep receiving."""
    connections = {"n": 0}
    received_subs = []

    async def server(ws):
        connections["n"] += 1
        for _ in range(4):  # 2 symbols x (kline + deal)
            received_subs.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"channel": "rs.sub.deal", "data": "success"}))
        await ws.send(json.dumps(deal(1, 7)))
        if connections["n"] == 1:
            await ws.close()  # simulate a dropped connection
            return
        await asyncio.sleep(5)

    async with websockets.serve(server, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        st = LiveMarketState(["BTCUSDT.P", "ETHUSDT.P"])
        stream = MexcFuturesStream(st, url=f"ws://127.0.0.1:{port}", ping_seconds=1)
        task = asyncio.create_task(stream.run())
        for _ in range(100):
            if connections["n"] >= 2 and st.connected and st.symbols["BTCUSDT.P"].messages >= 2:
                break
            await asyncio.sleep(0.05)
        stream.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert connections["n"] >= 2, "client did not reconnect"
    assert st.reconnects >= 1
    assert st.symbols["BTCUSDT.P"].messages >= 2
    assert {"method": "sub.deal", "param": {"symbol": "BTC_USDT"}} in received_subs
