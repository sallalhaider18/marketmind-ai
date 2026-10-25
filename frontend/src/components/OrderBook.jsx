import { baseAsset, fmtNum, fmtPrice } from '../api.js'

const sizeDigits = (symbol) => (symbol.startsWith('BTC') ? 2 : 1)

/** Diverging bar: -1 (all asks) .. +1 (all bids). */
function Lean({ value, label }) {
  const v = value ?? 0
  const pct = Math.min(Math.abs(v), 1) * 50
  const tone = v >= 0 ? 'up' : 'down'
  return (
    <div className="lean">
      <span className="lean-label">{label}</span>
      <div className="lean-track" aria-hidden="true">
        <div className={`lean-fill ${tone}`}
             style={v >= 0 ? { left: '50%', width: `${pct}%` } : { right: '50%', width: `${pct}%` }} />
        <div className="lean-zero" />
      </div>
      <span className={`lean-value ${value == null ? '' : tone}`}>
        {value == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
      </span>
    </div>
  )
}

export function ImbalancePanel({ ob }) {
  const imb = ob?.imbalance
  return (
    <section className="panel">
      <h2>Order-book lean, within 0.25% of price</h2>
      <div className="lean-scale"><span className="down">More asks</span><span className="up">More bids</span></div>
      <Lean label="Now" value={imb?.now} />
      <Lean label="5 s" value={imb?.avg_5s} />
      <Lean label="15 s" value={imb?.avg_15s} />
      <Lean label="60 s" value={imb?.avg_60s} />
      <p className="footnote">Resting orders can be moved or cancelled at any moment, so a lean only counts when it lasts and executed trades agree.</p>
    </section>
  )
}

export function LadderPanel({ ob, symbol }) {
  const lad = ob?.ladder
  const m = ob?.metrics
  if (!lad?.bids?.length) return null
  const all = [...lad.asks, ...lad.bids]
  const max = Math.max(...all.map((r) => r.size), 1e-9)
  const large = new Set((ob.large_orders?.active ?? []).map((o) =>
    `${o.side}:${o.side === 'bid' ? Math.floor(o.price / lad.step + 1e-9) : Math.ceil(o.price / lad.step - 1e-9)}`))
  const d = sizeDigits(symbol)
  const row = (r, side) => {
    const bucket = Math.round(r.price / lad.step)
    const isLarge = large.has(`${side}:${bucket}`)
    return (
      <div className={`ladder-row ${side}`} key={`${side}${r.price}`}>
        <div className="ladder-bar" style={{ width: `${(r.size / max) * 100}%` }} />
        <span className="ladder-price">{fmtPrice(r.price, symbol)}</span>
        <span className="ladder-size">{isLarge && <span className="large-dot" title="Large order in this range" />}{fmtNum(r.size, d)}</span>
      </div>
    )
  }
  return (
    <section className="panel">
      <h2>Depth, grouped by {fmtPrice(lad.step, symbol).replace(/\.0+$/, '')}</h2>
      <div className="ladder" role="table" aria-label="Order book depth">
        {[...lad.asks].reverse().map((r) => row(r, 'ask'))}
        <div className="ladder-spread">
          Spread {m?.spread != null ? fmtNum(m.spread, symbol.startsWith('BTC') ? 1 : 2) : '—'}
          {m?.spread_bps != null && ` (${m.spread_bps.toFixed(2)} bp)`}
        </div>
        {lad.bids.map((r) => row(r, 'bid'))}
      </div>
      <p className="footnote">Size in {baseAsset(symbol)}. Each row adds up all orders within that price range.</p>
    </section>
  )
}

const STATUS = {
  new: 'Just appeared',
  holding: 'Holding',
  growing: 'Growing',
  shrinking: 'Shrinking',
  absorbing: 'Absorbing trades',
  filled: 'Price reached it',
  pulled: 'Removed before price reached it',
}

function age(s) {
  return s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`
}

export function LargeOrdersPanel({ ob, symbol }) {
  const lo = ob?.large_orders
  const d = sizeDigits(symbol)
  const asset = baseAsset(symbol)
  return (
    <section className="panel">
      <h2>Large visible orders, nearest to price</h2>
      {!lo?.active?.length ? (
        <p className="muted">None right now.</p>
      ) : (
        <ul className="orders">
          {[...lo.active].sort((a, b) => Math.abs(a.distance_pct) - Math.abs(b.distance_pct)).slice(0, 6).map((o) => (
            <li key={`${o.side}${o.price}`} className={o.side === 'bid' ? 'up' : 'down'}>
              <span className="order-main">
                {o.side === 'bid' ? 'Bid' : 'Ask'} {fmtPrice(o.price, symbol)}
                <span className="order-size">{fmtNum(o.size, d)} {asset}</span>
              </span>
              <span className="order-meta">
                {STATUS[o.status]}, {age(o.age_s)}, {Math.abs(o.distance_pct).toFixed(2)}% from price
                {o.traded_into > 0 && `, ${fmtNum(o.traded_into, d)} traded into it`}
              </span>
            </li>
          ))}
        </ul>
      )}
      {lo?.recent?.length > 0 && (
        <>
          <h3>Recently gone</h3>
          <ul className="orders recent">
            {lo.recent.slice(0, 4).map((o, i) => (
              <li key={`${o.side}${o.price}${i}`}>
                <span className="order-main">
                  {o.side === 'bid' ? 'Bid' : 'Ask'} {fmtPrice(o.price, symbol)}
                  <span className="order-size">{fmtNum(o.max_size, d)} {asset}</span>
                </span>
                <span className="order-meta">{STATUS[o.status]} after {age(o.age_s)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      {lo?.threshold != null && (
        <p className="footnote">
          Large means at least {fmtNum(lo.threshold, d)} {asset} at one price, 8× the typical level
          ({fmtNum(lo.typical_level, d + 1)} {asset}) within {(lo.band_pct * 100).toFixed(2)}% of price.
          Alerts show only the strongest bid and ask of at least 100 BTC / 1,000 ETH that held 30 s. This shows what is visible, not who placed it.
        </p>
      )}
    </section>
  )
}

export function AlertsStrip({ ob }) {
  const alerts = ob?.alerts ?? []
  return (
    <section className="alerts" aria-label="Alerts" aria-live="polite">
      <h2>Alerts</h2>
      {alerts.length === 0 ? (
        <p className="muted">No alerts yet. Only the strongest bid and ask near the price are reported (at least 100 BTC / 1,000 ETH, held 30 s), then whether they hold, break or were fake.</p>
      ) : (
        <ol>
          {alerts.slice(0, 6).map((a) => (
            <li key={a.id}>
              <time>{new Date(a.time).toLocaleTimeString('en-GB')}</time>
              <span className={`alert-mark ${a.tone}`} aria-hidden="true" />
              <span>{a.text}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
