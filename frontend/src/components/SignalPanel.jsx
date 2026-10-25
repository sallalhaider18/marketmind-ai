import { fmtNum, fmtPrice } from '../api.js'

const ukTime = (sec) => new Date(sec * 1000).toLocaleTimeString('en-GB', {
  timeZone: 'Europe/London', hour: '2-digit', minute: '2-digit' })
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`)
const sgn = (v, d = 2) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtNum(v, d)}`)

const LEAN = {
  bounce: 'BOUNCE candidate → long', breakdown: 'BREAKDOWN candidate → short',
  rejection: 'REJECTION candidate → short', breakout: 'BREAKOUT candidate → long', uncertain: 'uncertain → no trade',
}
const FATE = { held: 'wall holding', absorbing: 'wall absorbing (traded into, still there)',
  consumed: 'wall consumed by trades', pulled: 'wall pulled (cancelled)', none: 'no big wall' }

function checks(branch) {
  const reversal = branch === 'bounce' || branch === 'rejection'
  return [
    ['wall', reversal ? 'Wall defending the level' : 'No wall left defending'],
    ['imbalance', 'Imbalance 0.1% + 0.25% (60 s)'],
    ['flow', 'Market orders (60 s)'],
    ['cvd', reversal ? 'CVD last minute' : 'CVD last 5 minutes'],
    ['price', reversal ? '5-min close back on its side' : 'Two 5-min closes through'],
  ]
}

function value(key, v, asset, branch) {
  switch (key) {
    case 'wall': return v.wall_coins == null ? 'none' : `${fmtNum(v.wall_coins, 1)} ${asset}, ${Math.round(v.wall_age ?? 0)} s`
    case 'imbalance': return `${sgn(v.imb10)} / ${sgn(v.imb25)}`
    case 'flow': return pct(v.flow_share)
    case 'cvd': {
      const c = branch === 'bounce' || branch === 'rejection' ? v.cvd_1m : v.cvd_5m
      return c == null ? '—' : `${sgn(c, 1)} ${asset}`
    }
    case 'price': return v.candle_close == null ? '—' : fmtNum(v.candle_close, 1)
    default: return ''
  }
}

function Setup({ s, symbol, asset }) {
  const up = s.trade === 'long'
  return (
    <div className={`plan-card ${up ? 'long' : 'short'}`}>
      <p className="plan-headline">
        {s.kind === 'support' ? 'Support' : 'Resistance'} <strong>{fmtPrice(s.level, symbol)}</strong>
        {s.source && <span className="muted-inline"> ({s.source})</span>}
        {' — '}{s.touched ? `touched, ${s.checks} five-min checks` : 'waiting for a touch'}
      </p>
      <p className="setup-lean">
        Leaning: <strong className={s.leaning === 'uncertain' ? '' : (up ? 'up' : 'down')}>{LEAN[s.leaning]}</strong>
        <span className="muted-inline"> · {FATE[s.fate] ?? s.fate}</span>
      </p>
      <ul className="checks">
        {checks(s.branch).map(([k, label]) => (
          <li key={k} className={s.ok[k] ? 'pass' : 'fail'}>
            <span aria-hidden="true">{s.ok[k] ? '✓' : '✗'}</span>
            <span>{label}</span>
            <span className="check-val">{value(k, s.values, asset, s.branch)}</span>
          </li>
        ))}
      </ul>
      {s.off.length > 0 && <p className="footnote">Switched off: {s.off.join(', ')} (30-min close).</p>}
    </div>
  )
}

function BySetup({ by }) {
  if (!by) return null
  const cell = (x) => (x?.trades ? `${x.trades} · ${sgn(x.expectancy_r)} R` : '—')
  return (
    <table className="sig-stats by-setup">
      <thead><tr><th scope="col">By setup type</th><th scope="col">A</th><th scope="col">B</th></tr></thead>
      <tbody>
        {Object.entries(by).map(([name, r]) => (
          <tr key={name}><th scope="row">{name}</th><td>{cell(r.A)}</td><td>{cell(r.B)}</td></tr>
        ))}
      </tbody>
    </table>
  )
}

function StatsTable({ stats }) {
  const cols = ['A', 'B']
  const rows = [
    ['Closed trades', (s) => s.trades ?? 0],
    ['Win rate', (s) => (s.trades ? pct(s.win_rate) : '—')],
    ['Expectancy (R, net)', (s) => (s.trades ? sgn(s.expectancy_r) : '—')],
    ['Total R', (s) => (s.trades ? sgn(s.total_r, 1) : '—')],
    ['Profit factor', (s) => (s.profit_factor == null ? '—' : fmtNum(s.profit_factor, 2))],
    ['Max drawdown (R)', (s) => (s.trades ? fmtNum(s.max_drawdown_r, 1) : '—')],
    ['Limit fill rate', (s, k) => (k === 'B' && s.fill_rate != null ? pct(s.fill_rate) : '—')],
    ['Paper account', (s) => `$${fmtNum(s.equity, 0)}`],
  ]
  return (
    <table className="sig-stats">
      <thead><tr><th scope="col" /><th scope="col">A: confirmed, market</th><th scope="col">B: limit at retest</th></tr></thead>
      <tbody>
        {rows.map(([label, f]) => (
          <tr key={label}><th scope="row">{label}</th>{cols.map((k) => <td key={k}>{f(stats[k] ?? {}, k)}</td>)}</tr>
        ))}
      </tbody>
    </table>
  )
}

export function SignalPanel({ sig, symbol }) {
  if (!sig) {
    return (
      <section className="panel">
        <h2>Order-flow signals (A / B)</h2>
        <p className="muted">Waiting for the order book…</p>
      </section>
    )
  }
  const asset = symbol.replace('USDT.P', '')
  const n = sig.nearest
  const distance = (lv) => (lv && n?.mid ? `${((lv / n.mid - 1) * 100).toFixed(2)}%` : '')
  return (
    <section className="panel" aria-live="polite">
      <h2>Order-flow signals (A / B)</h2>
      {sig.setups.length ? sig.setups.map((s) => <Setup key={s.kind} s={s} symbol={symbol} asset={asset} />) : (
        <p className="muted">
          No setup: price is not within {(sig.rules.near_pct * 100).toFixed(2)}% of a strong level.
          {n && <> Nearest support {n.support ? `${fmtPrice(n.support, symbol)} (${distance(n.support)})` : '—'},
            resistance {n.resistance ? `${fmtPrice(n.resistance, symbol)} (+${distance(n.resistance)})` : '—'}.</>}
        </p>
      )}

      {sig.positions.length > 0 && (
        <>
          <h3>Paper positions</h3>
          <ul className="plan-history">
            {sig.positions.map((p) => (
              <li key={p.strategy}>
                <span className={p.side === 'long' ? 'up' : 'down'}>
                  {p.strategy}: {p.side === 'long' ? 'Long' : 'Short'} {p.status === 'pending' ? 'limit' : ''} {fmtPrice(p.entry, symbol)}
                </span>
                <span className="muted-inline">SL {fmtPrice(p.stop, symbol)} · TP {fmtPrice(p.target, symbol)}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <h3>Results so far (paper, after fees)</h3>
      <StatsTable stats={sig.stats} />
      <BySetup by={sig.stats.by_setup} />

      {sig.level_tests?.length > 0 && (
        <>
          <h3>Level tests (what happened within 60 min of a touch)</h3>
          <ul className="plan-history">
            {sig.level_tests.map((x, i) => (
              <li key={i}>
                <span className={x.outcome === 'broke' ? 'down' : x.outcome === 'held' ? 'up' : ''}>
                  {ukTime(x.time)} {x.kind} {fmtPrice(x.level, symbol)}: {x.outcome}
                </span>
                <span className="muted-inline">{x.max_beyond_pct.toFixed(2)}% through · {x.max_back_pct.toFixed(2)}% back · wall {x.wall_fate}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {sig.evaluations.length > 0 && (
        <>
          <h3>Latest checks</h3>
          <ul className="plan-history">
            {sig.evaluations.slice(0, 5).map((e, i) => (
              <li key={i}>
                <span className={e.passed ? (e.side === 'long' ? 'up' : 'down') : ''}>
                  {ukTime(e.time)} {e.kind} {fmtPrice(e.level, symbol)}
                </span>
                <span className="muted-inline">{e.reason}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="footnote">
        Four setups: support bounce / breakdown, resistance rejection / breakout. Reversals need a wall ≥ {fmtNum(sig.rules.wall_min, 0)} {asset}{' '}
        defending the level (shrinking by trades is fine, cancelling is not); breaks need it gone and two 5-min closes through.
        Both need imbalance ≥ {sig.rules.imb_min}, market orders ≥ {pct(sig.rules.flow_share)} and CVD in the trade direction, and
        reward:risk ≥ {sig.rules.rr_min} after costs. A enters at once; B places a limit at the retest. Paper only. Logs:
        reports/signal_evaluations.csv, signal_trades.csv, level_tests.csv.
      </p>
    </section>
  )
}
