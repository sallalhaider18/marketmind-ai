import { fmtNum, fmtPrice } from '../api.js'

const TONE_CLASS = { up: 'up', down: 'down', warn: 'warn', neutral: '' }
const time = (iso) => new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

function Spark({ cvd }) {
  if (!cvd || cvd.length < 3) return <p className="muted">The CVD line appears after a few minutes of trades.</p>
  const W = 300, H = 70, n = cvd.length
  const line = (key) => {
    const vals = cvd.map((p) => p[key])
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1
    return cvd.map((p, i) => `${(i / (n - 1)) * W},${H - 4 - ((p[key] - lo) / span) * (H - 8)}`).join(' ')
  }
  const rising = cvd[n - 1].cvd >= cvd[Math.max(0, n - 16)].cvd
  return (
    <figure className="cvd">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
           aria-label={`CVD ${rising ? 'rising' : 'falling'} over the last ${n} minutes`}>
        <polyline points={line('price')} fill="none" stroke="#9AA4AE" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
        <polyline points={line('cvd')} fill="none" stroke={rising ? 'var(--up)' : 'var(--down)'} strokeWidth="2"
                  vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption>
        <span className={rising ? 'up' : 'down'}>━ CVD (market buys − sells)</span>
        <span className="muted-inline">━ price</span>
        <span className="muted-inline">last {n} min</span>
      </figcaption>
    </figure>
  )
}

function WindowRow({ w, asset }) {
  if (!w) return null
  const r = w.reading
  return (
    <div className="flow-window">
      <div className="flow-window-head">
        <strong>Last {w.minutes} min</strong>
        <span className={`chip ${TONE_CLASS[r.tone]}`}>{r.label}</span>
      </div>
      <div className="flow-window-nums">
        <span className="up">Bought {fmtNum(w.bought, 1)}</span>
        <span className="down">Sold {fmtNum(w.sold, 1)}</span>
        <span>{asset}, price {w.price_change_pct == null ? '—' : `${w.price_change_pct >= 0 ? '+' : ''}${w.price_change_pct.toFixed(2)}%`}</span>
      </div>
      <p className="flow-text">{r.text}</p>
    </div>
  )
}

export function OrderFlowPanel({ flow, symbol }) {
  if (!flow) {
    return (
      <section className="panel">
        <h2>Market orders: who is pushing?</h2>
        <p className="muted">Waiting for live trades…</p>
      </section>
    )
  }
  const o = flow.overall
  return (
    <section className="panel flow-panel" aria-live="polite">
      <h2>Market orders: who is pushing?</h2>
      <div className={`flow-verdict ${TONE_CLASS[o.tone]}`}>
        <div className="flow-verdict-summary">{o.summary}</div>
        <div className="flow-verdict-label">{o.label}</div>
      </div>

      <WindowRow w={flow.windows[0]} asset={flow.asset} />
      <WindowRow w={flow.windows[1]} asset={flow.asset} />

      <h3>CVD vs price</h3>
      <Spark cvd={flow.cvd} />

      <h3>Big market orders (≥ {flow.big_order_size} {flow.asset})</h3>
      {flow.big_orders.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <ul className="big-orders">
          {flow.big_orders.slice(0, 8).map((b, i) => (
            <li key={i}>
              <span className={b.side === 'buy' ? 'up' : 'down'}>{b.side === 'buy' ? 'BUY' : 'SELL'} {fmtNum(b.size, 1)} {flow.asset}</span>
              <span>@ {fmtPrice(b.price, symbol)}</span>
              <time>{time(b.time)}</time>
            </li>
          ))}
        </ul>
      )}

      <details className="flow-help">
        <summary>How to read this</summary>
        <ul>
          <li><b>UP / DOWN (strong):</b> the last 5 and 15 minutes agree. Real buyers (or sellers) are pushing.</li>
          <li><b>Weak rise / weak drop:</b> price moved, but market orders went the other way. Often reverses. Don't chase it.</li>
          <li><b>Sellers absorbed:</b> heavy selling, price not falling: big buyers are defending. Bullish.
            <b> Buyers absorbed</b> is the opposite. Bearish.</li>
          <li><b>CVD:</b> price and CVD rising together = healthy move. Price up but CVD flat or down = weak.</li>
          <li><b>Best use:</b> when price reaches your planned level, take the trade if this panel agrees with it, skip it if it says the opposite.</li>
        </ul>
      </details>
      <p className="footnote">Counts only trades seen since the server started{flow.since ? ` (${time(flow.since)})` : ''}. Shows who is in control now, not where price will be in hours.</p>
    </section>
  )
}
