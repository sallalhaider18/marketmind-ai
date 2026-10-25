import { useState } from 'react'
import { fmtNum, fmtPrice } from '../api.js'

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`)
const ukTime = (iso) => (iso ? new Date(iso).toLocaleTimeString('en-GB', {
  timeZone: 'Europe/London', hour: '2-digit', minute: '2-digit' }) : '')

async function send(method, symbol, price) {
  const url = method === 'DELETE' ? `/api/levels/${symbol}?price=${price}` : `/api/levels/${symbol}`
  const r = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify({ price }) : undefined,
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

function dirOf(kind, verdict) {
  // breakout at resistance / bounce at support -> LONG; breakdown at support / rejection at resistance -> SHORT
  const v = verdict.replace('_lean', '')
  if (v !== 'breakout' && v !== 'reversal') return null
  return (kind === 'resistance') === (v === 'breakout') ? 'LONG' : 'SHORT'
}

function Verdict({ w }) {
  const v = w.v2
  if (!v) return null
  const dir = dirOf(w.kind, v.verdict)
  const type = v.verdict.startsWith('breakout') ? 'BREAKOUT' : 'REVERSAL'
  const icon = dir === 'LONG' ? '🟢' : '🔴'
  let main, cls
  if (v.verdict === 'insufficient') { main = '⚠️ INSUFFICIENT DATA — no signal'; cls = 'none' }
  else if (v.verdict === 'conflicting') { main = '🟠 CONFLICTING — both sides strong, no trade'; cls = 'none' }
  else if (v.verdict === 'inconclusive') { main = '⚪ NO TRADE — wait'; cls = 'none' }
  else if (v.verdict.endsWith('_lean')) { main = `${icon} GET READY: possible ${dir} — ${type}`; cls = `${dir === 'LONG' ? 'long' : 'short'} moderate` }
  else { main = `${icon} ${dir} — ${type} (STRONG)`; cls = `${dir === 'LONG' ? 'long' : 'short'} high` }
  return (
    <div className={`sr-signal ${cls}`}>
      <p className="sr-signal-main">{main}</p>
      <p className="sr-signal-why">{v.reason}</p>
      {v.breakout_score != null && (
        <div className="sr-scores" aria-label={`Breakout score ${v.breakout_score}, reversal score ${v.reversal_score}`}>
          {[['Breakout', v.breakout_score], ['Reversal', v.reversal_score]].map(([k, x]) => (
            <div key={k} className="sr-score">
              <span>{k}</span>
              <span className="sr-bar"><span style={{ width: `${Math.max(0, Math.min(100, x))}%` }}
                className={x >= 75 ? 'strong' : x >= 60 ? 'lean' : ''} /></span>
              <strong>{Math.round(x)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function baselineText(w) {
  const dir = dirOf(w.kind, w.verdict)
  if (!dir) return 'no signal'
  return `${dir} ${w.verdict === 'breakout' ? 'breakout' : 'reversal'} (${w.strength})`
}

function Level({ w, symbol, asset }) {
  const r = w.readings
  const attackers = w.kind === 'resistance' ? 'buying' : 'selling'
  const defence = w.kind === 'resistance' ? 'Ask' : 'Bid'
  const w60 = w.windows['60s']
  return (
    <div className={`plan-card ${w.kind === 'resistance' ? 'short' : 'long'}`}>
      <p className="plan-headline">
        {w.kind === 'resistance' ? 'Resistance' : 'Support'} <strong>{fmtPrice(w.level, symbol)}</strong>
        <span className="muted-inline"> {w.distance_pct >= 0 ? '+' : ''}{w.distance_pct.toFixed(2)}% away
          {w.origin === 'auto' ? ' · auto level' : ''}{w.visiting ? ' · VISIT in progress' : ''}</span>
      </p>
      <Verdict w={w} />
      {r ? (
        <dl className="sr-readings">
          <div className="row"><dt>{defence} liquidity consumption</dt><dd>{r.consumption} <span className="muted-inline">{pct(r.consumption_pct)}/min</span></dd></div>
          <div className="row"><dt>Aggressive {attackers}</dt><dd>{r.attack} <span className="muted-inline">{pct(r.attack_share)}</span></dd></div>
          <div className="row"><dt>{defence} replenishment</dt><dd>{r.replenishment} <span className="muted-inline">{r.refill_ratio == null ? '' : `${fmtNum(r.refill_ratio, 2)}×`}</span></dd></div>
          <div className="row"><dt>CVD (60 s)</dt><dd>{r.cvd} <span className="muted-inline">{fmtNum(r.cvd_value, 1)} {asset}</span></dd></div>
          <div className="row"><dt>Cancelled (pulled)</dt><dd>{pct(r.cancelled_pct)}</dd></div>
          <div className="row"><dt>Price progress (60 s)</dt><dd>{r.progress_pct >= 0 ? '+' : ''}{r.progress_pct.toFixed(3)}% toward/through</dd></div>
          <div className="row"><dt>Price response vs normal</dt><dd>{w.v2?.impact_ratio == null ? 'learning (first 10 min)'
            : `${fmtNum(w.v2.impact_ratio, 2)}× ${w.v2.impact_ratio < 0.5 ? '(absorbed)' : w.v2.impact_ratio > 1.5 ? '(moving easily)' : ''}`}</dd></div>
          <div className="row"><dt>Same-price liquidity survived</dt><dd>{pct(w.v2?.survival)}</dd></div>
          <div className="row"><dt>Attack held (last 15 s)</dt><dd>{pct(w.v2?.persistence)}</dd></div>
          {w.v2?.failed_cross && <div className="row"><dt>Failed cross</dt><dd className="warn-text">yes: poked through and came back</dd></div>}
        </dl>
      ) : <p className="muted">Collecting the first 60 seconds…</p>}
      <p className="sr-split">
        <span>Displayed now: <strong>{fmtNum(w.depth10, 1)}</strong> {asset} within 0.1%, {fmtNum(w.depth25, 1)} within 0.25%</span>
        {w60 && <span>Executed into it (60 s): <strong>{fmtNum(w60.executed, 1)}</strong> {asset}</span>}
      </p>
      <p className="footnote">Baseline rules (for comparison): {baselineText(w)}.</p>
    </div>
  )
}

export function SRAnalyzerPanel({ sr, symbol }) {
  const [input, setInput] = useState('')
  const [error, setError] = useState(null)
  const asset = symbol.replace('USDT.P', '')
  if (!sr) {
    return (
      <section className="panel"><h2>S/R order-flow analyzer</h2>
        <p className="muted">Waiting for the order book…</p></section>
    )
  }
  const add = async (e) => {
    e.preventDefault()
    const price = parseFloat(input.replace(/,/g, ''))
    if (!(price > 0)) { setError('Type a price, e.g. 81500'); return }
    try { await send('POST', symbol, price); setInput(''); setError(null) } catch (err) { setError(String(err)) }
  }
  const remove = async (p) => { try { await send('DELETE', symbol, p) } catch (err) { setError(String(err)) } }
  const sc = sr.scorecard
  return (
    <section className="panel" aria-live="polite">
      <h2>S/R order-flow analyzer</h2>
      <form className="sr-add" onSubmit={add}>
        <label htmlFor="sr-level">My {asset} level</label>
        <input id="sr-level" inputMode="decimal" placeholder="e.g. 81500" value={input}
               onChange={(e) => setInput(e.target.value)} />
        <button type="submit">Add</button>
      </form>
      {error && <p className="sr-error" role="alert">{error}</p>}
      {sr.origin === 'mine' ? (
        <ul className="sr-levels">
          {sr.levels.map((p) => (
            <li key={p}>{fmtPrice(p, symbol)}
              <button type="button" aria-label={`Remove ${p}`} onClick={() => remove(p)}>×</button></li>
          ))}
        </ul>
      ) : (
        <p className="muted">No levels of your own yet, so it is watching the strong daily/4H levels. Add yours above.</p>
      )}

      {sr.watched.length ? sr.watched.map((w) => <Level key={w.level} w={w} symbol={symbol} asset={asset} />) : (
        <p className="muted">No level within {(sr.rules.watch_pct * 100).toFixed(0)}% of price. Analysis starts when price gets closer.</p>
      )}

      <h3>Track record (visits within {(sr.rules.visit_pct * 100).toFixed(1)}% of a level)</h3>
      {sc.visits ? (
        <dl>
          <div className="row"><dt>Independent visits</dt><dd>{sc.visits}{sc.overlapping_excluded ? ` (+${sc.overlapping_excluded} overlapping, not counted)` : ''}</dd></div>
          <div className="row"><dt>Base rate</dt><dd>broke {pct(sc.broke_rate)} · held {pct(sc.held_rate)} · unresolved {pct(sc.unresolved_rate)}</dd></div>
          <div className="row"><dt>Scored: breakout calls</dt><dd>{sc.v2.breakout_calls} · then broke {pct(sc.v2.breakout_right)}</dd></div>
          <div className="row"><dt>Scored: reversal calls</dt><dd>{sc.v2.reversal_calls} · then held {pct(sc.v2.reversal_right)}</dd></div>
          <div className="row"><dt>Baseline: breakout calls</dt><dd>{sc.breakout_calls} · then broke {pct(sc.breakout_right)}</dd></div>
          <div className="row"><dt>Baseline: reversal calls</dt><dd>{sc.reversal_calls} · then held {pct(sc.reversal_right)}</dd></div>
        </dl>
      ) : <p className="muted">No finished visits yet. Each visit is judged 60 min after it starts.</p>}
      {sr.recent_visits.length > 0 && (
        <ul className="plan-history">
          {sr.recent_visits.map((v, i) => (
            <li key={i}>
              <span>{ukTime(v.time)} {v.kind} {fmtPrice(v.level, symbol)}</span>
              <span className="muted-inline">scored: {v.call2 || '—'} · baseline: {v.call} → <strong>{v.outcome}</strong></span>
            </li>
          ))}
        </ul>
      )}
      {sr.clock_offset_s != null && Math.abs(sr.clock_offset_s) >= 2 && (
        <p className="footnote">Your computer's clock is about {Math.abs(sr.clock_offset_s).toFixed(0)} s
          {sr.clock_offset_s > 0 ? ' ahead of' : ' behind'} MEXC's. The analyzer allows for this, but you can fix it in
          Windows: Settings → Time &amp; language → Date &amp; time → Sync now.</p>
      )}
      <p className="footnote">Order book and executed trades only, no candles or indicators. Consumed = traded away;
        cancelled = pulled without trades; replenished = new orders or hidden refills. Outcome = price moved
        {' '}{(sr.rules.outcome_pct * 100).toFixed(1)}% through (broke) or back (held) first, within 60 min. A verdict is
        evidence to weigh, not a prediction with proven accuracy.</p>
    </section>
  )
}
