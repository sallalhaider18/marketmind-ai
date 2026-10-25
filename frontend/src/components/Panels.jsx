import { baseAsset, fmtNum, fmtTime } from '../api.js'

function Row({ label, children, tone }) {
  return (
    <div className="row">
      <dt>{label}</dt>
      <dd className={tone ?? ''}>{children}</dd>
    </div>
  )
}

export function FeedPanel({ feed, offline, ob }) {
  let state = 'Waiting for data'
  let tone = 'warn'
  if (offline) state = 'Backend offline'
  else if (feed && !feed.connected) state = 'Reconnecting'
  else if (feed?.stale) state = 'Stale'
  else if (feed) { state = 'Live'; tone = 'ok' }

  const h = feed?.history
  return (
    <section className="panel">
      <h2>Data feed</h2>
      <dl>
        <Row label="MEXC stream" tone={tone}><span className={`dot ${tone}`} aria-hidden="true" />{state}</Row>
        <Row label="Last update">
          {feed?.seconds_since_update == null ? '—' : `${feed.seconds_since_update.toFixed(1)} s ago`}
        </Row>
        <Row label="Order book" tone={ob?.status === 'live' ? 'ok' : 'warn'}>
          {ob == null ? '—' : ob.status === 'live' ? 'In sync' : ob.status === 'stale' ? 'Stale' : 'Syncing…'}
        </Row>
        <Row label="Book self-check">
          {ob?.health?.refreshes ? `${ob.health.refreshes} done, ${Math.round((ob.health.last_drift ?? 0) * 100)}% off` : 'every 2 min'}
        </Row>
        <Row label="Reconnects / resyncs">{feed?.reconnects ?? '—'} / {ob?.health?.resyncs ?? '—'}</Row>
        <Row label="Stored 1m candles">{h ? h.stored_candles.toLocaleString('en-GB') : '—'}</Row>
        <Row label="History synced">{fmtTime(h?.last_sync)}</Row>
      </dl>
      {h?.sync_error && <p className="panel-error">History sync failed: {h.sync_error}</p>}
      {feed?.last_error && !feed.connected && <p className="panel-error">{feed.last_error}</p>}
      {ob && ob.status === 'syncing' && ob.health?.last_resync_reason && ob.health.last_resync_reason !== 'new connection' && (
        <p className="panel-error">Order book: {ob.health.last_resync_reason}</p>
      )}
    </section>
  )
}

export function FlowPanel({ feed, symbol }) {
  const flow = feed?.trade_flow_60s
  const size = feed?.history?.contract_size
  const asset = size ? baseAsset(symbol) : 'contracts'
  const conv = (v) => (size ? v * size : v)
  const digits = size ? 2 : 0
  const ratio = flow?.buy_ratio

  return (
    <section className="panel">
      <h2>Aggressive buying vs selling, last 60 s</h2>
      {ratio == null ? (
        <p className="muted">No trades in the last 60 seconds yet.</p>
      ) : (
        <>
          <div className="flow-bar" role="img"
               aria-label={`Buyers ${Math.round(ratio * 100)} percent, sellers ${Math.round((1 - ratio) * 100)} percent`}>
            <div className="flow-buy" style={{ width: `${ratio * 100}%` }} />
            <div className="flow-mid" />
          </div>
          <div className="flow-legend">
            <span className="up">Buyers {Math.round(ratio * 100)}%</span>
            <span className="down">Sellers {Math.round((1 - ratio) * 100)}%</span>
          </div>
          <dl>
            <Row label="Bought" tone="up">{fmtNum(conv(flow.buy_contracts), digits)} {asset}</Row>
            <Row label="Sold" tone="down">{fmtNum(conv(flow.sell_contracts), digits)} {asset}</Row>
            <Row label="Net (delta)" tone={flow.delta_contracts >= 0 ? 'up' : 'down'}>
              {flow.delta_contracts >= 0 ? '+' : ''}{fmtNum(conv(flow.delta_contracts), digits)} {asset}
            </Row>
            <Row label="Trades">{flow.trades.toLocaleString('en-GB')}</Row>
          </dl>
        </>
      )}
      <p className="footnote">Measured from executed trades: who crossed the spread. It describes pressure, not intent.</p>
    </section>
  )
}
