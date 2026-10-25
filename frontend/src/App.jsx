import { useCallback, useEffect, useRef, useState } from 'react'
import PriceChart from './components/PriceChart.jsx'
import { FeedPanel } from './components/Panels.jsx'
import { OrderFlowPanel } from './components/OrderFlowPanel.jsx'
import { AlertsStrip, ImbalancePanel, LadderPanel, LargeOrdersPanel } from './components/OrderBook.jsx'
import { PlanPanel } from './components/PlanPanel.jsx'
import { SignalPanel } from './components/SignalPanel.jsx'
import { SRAnalyzerPanel } from './components/SRAnalyzerPanel.jsx'
import { fmtPrice, getJSON, useLiveFeed } from './api.js'

const remember = (key, fallback) => {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}
const save = (key, value) => {
  try { localStorage.setItem(key, value) } catch { /* storage unavailable */ }
}

function SymbolTab({ symbol, active, onSelect, feed, bar }) {
  const price = feed?.last_price
  const dir = bar ? (bar.close >= bar.open ? 'up' : 'down') : ''
  return (
    <button className={`symbol-tab ${active ? 'is-active' : ''}`} onClick={() => onSelect(symbol)}
            aria-pressed={active}>
      <span className="symbol-name">{symbol}</span>
      <span className={`symbol-price ${dir}`}>{fmtPrice(price, symbol)}</span>
    </button>
  )
}

export default function App() {
  const [config, setConfig] = useState({ symbols: ['BTCUSDT.P', 'ETHUSDT.P'], intervals: ['30m', '4h', '1d'] })
  // (support/resistance lines show on every timeframe)
  const [symbol, setSymbol] = useState(() => remember('mm.symbol', 'BTCUSDT.P'))
  const [interval, setIntervalState] = useState(() => {
    const saved = remember('mm.interval', '4h')
    return ['30m', '4h', '1d'].includes(saved) ? saved : '4h'
  })
  const [loadError, setLoadError] = useState(null)
  const [loading, setLoading] = useState(true)
  const chart = useRef(null)
  const request = useRef(0)

  useEffect(() => { getJSON('/api/dashboard/config').then(setConfig).catch(() => {}) }, [])

  const selectSymbol = (s) => { setSymbol(s); save('mm.symbol', s) }
  const selectInterval = (i) => { setIntervalState(i); save('mm.interval', i) }

  // Load history whenever symbol or timeframe changes
  useEffect(() => {
    const id = ++request.current
    setLoading(true)
    getJSON(`/api/dashboard/chart/${symbol}?interval=${interval}&limit=1500`)
      .then((r) => {
        if (id !== request.current) return
        chart.current?.setBars(r.bars)
        setLoadError(r.bars.length ? null : 'No candles stored yet. The backend downloads them on start-up; this can take a minute.')
      })
      .catch((e) => id === request.current && setLoadError(`Could not load the chart (${e.message}). Is the backend running?`))
      .finally(() => id === request.current && setLoading(false))
  }, [symbol, interval])

  const onLive = useCallback((msg) => {
    if (msg.symbol === symbol && msg.interval === interval) chart.current?.updateBars(msg.bars)
  }, [symbol, interval])

  const active = useLiveFeed(symbol, interval, onLive)
  const other = config.symbols.find((s) => s !== symbol)
  const background = useLiveFeed(other ?? symbol, '30m')

  const feed = active.message?.feed
  const ob = active.message?.orderbook

  const plan = active.message?.plan
  const sig = active.message?.signals
  const sr = active.message?.sr

  // the current trade plan on the chart (support/resistance lines are not drawn;
  // the plan still uses them for entry, target and stop)
  useEffect(() => {
    chart.current?.setLevels([], symbol)
    chart.current?.setPlan(plan?.active ?? null, symbol)
  }, [plan, symbol, loading])
  const lastBar = active.message?.bars?.at(-1)
  const offline = !active.backendOnline
  const stale = offline || (feed ? feed.stale : false)
  const staleText = offline
    ? 'Backend is not reachable. Start it with: uvicorn app.main:app'
    : !feed?.connected
      ? 'Exchange feed disconnected. Reconnecting automatically; prices on screen may be out of date.'
      : `No market data for ${Math.round(feed.seconds_since_update ?? 0)} s. Prices on screen may be out of date.`

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">MarketMind</div>
        <nav className="symbols" aria-label="Instrument">
          {config.symbols.map((s) => {
            const src = s === symbol ? active.message : background.message
            return <SymbolTab key={s} symbol={s} active={s === symbol} onSelect={selectSymbol}
                              feed={src?.feed} bar={src?.bars?.at(-1)} />
          })}
        </nav>
        <div className="intervals" role="group" aria-label="Timeframe">
          {config.intervals.map((i) => (
            <button key={i} className={i === interval ? 'is-active' : ''} aria-pressed={i === interval}
                    onClick={() => selectInterval(i)}>{i === '4h' ? '4H' : i === '1d' ? '1D' : i}</button>
          ))}
        </div>
      </header>

      <main className="workspace">
        <section className={`chart-area ${stale ? 'is-stale' : ''}`} aria-label={`${symbol} ${interval} chart`}>
          {stale && (feed || offline) && <div className="stale-band" role="status">{staleText}</div>}
          <PriceChart ref={chart} symbol={symbol} />
          {loading && <div className="chart-note">Loading {symbol} {interval}…</div>}
          {!loading && loadError && <div className="chart-note">{loadError}</div>}
          <AlertsStrip ob={ob} />
        </section>

        <aside className="rail">
          <SRAnalyzerPanel sr={sr} symbol={symbol} />
          <PlanPanel plan={plan} symbol={symbol} flow={ob?.flow} />
          <SignalPanel sig={sig} symbol={symbol} />
          <OrderFlowPanel flow={ob?.flow} symbol={symbol} />
          <FeedPanel feed={feed} offline={offline} ob={ob} />
          <ImbalancePanel ob={ob} />
          <LadderPanel ob={ob} symbol={symbol} />
          <LargeOrdersPanel ob={ob} symbol={symbol} />
        </aside>
      </main>
    </div>
  )
}
