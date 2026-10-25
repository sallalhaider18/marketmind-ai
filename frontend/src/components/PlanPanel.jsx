import { fmtNum, fmtPrice } from '../api.js'

const ukTime = (iso) => (iso ? new Date(iso).toLocaleTimeString('en-GB', {
  timeZone: 'Europe/London', hour: '2-digit', minute: '2-digit' }) : '—')
const ukDayTime = (iso) => (iso ? new Date(iso).toLocaleString('en-GB', {
  timeZone: 'Europe/London', weekday: 'short', hour: '2-digit', minute: '2-digit' }) : '—')

function Row({ label, children, tone }) {
  return (
    <div className="row"><dt>{label}</dt><dd className={tone ?? ''}>{children}</dd></div>
  )
}

export function PlanPanel({ plan, symbol, flow }) {
  const a = plan?.active
  const acct = plan?.account
  const long = a?.side === 'long'
  const sessions = plan?.sessions?.length ? plan.sessions.join(' + ') : 'none open'
  return (
    <section className="panel plan-panel" aria-live="polite">
      <h2>Order-book trade plan</h2>
      <p className="plan-meta">
        Sessions: {sessions}. Next check {ukTime(plan?.next_check)} UK (every 30 min in Asia, London and New York).
      </p>

      {a ? (
        <div className={`plan-card ${long ? 'long' : 'short'}`}>
          <p className="plan-headline">
            {a.status === 'pending'
              ? <>Place a <strong>limit {long ? 'buy' : 'sell'}</strong> at <strong>{fmtPrice(a.entry, symbol)}</strong></>
              : <><strong>{long ? 'Long' : 'Short'}</strong> open from <strong>{fmtPrice(a.fill_price, symbol)}</strong></>}
          </p>
          <dl>
            <Row label="Entry (limit)">{fmtPrice(a.entry, symbol)}</Row>
            <Row label="Take profit" tone="up">{fmtPrice(a.target, symbol)}</Row>
            <Row label="Stop loss" tone="down">{fmtPrice(a.stop, symbol)}</Row>
            <Row label="Reward : risk">{fmtNum(a.rr, 2)} : 1</Row>
            <Row label="Status">
              {a.status === 'pending' ? `waiting, cancels ${ukDayTime(a.expires_at)} UK if unfilled`
                                      : `filled ${ukDayTime(a.filled_at)} UK`}
            </Row>
          </dl>
          {flow?.overall && (() => {
            const t = flow.overall.tone
            const agrees = (long && t === 'up') || (!long && t === 'down')
            const against = (long && t === 'down') || (!long && t === 'up')
            return (
              <p className={`plan-flow ${agrees ? 'up' : against ? 'down' : ''}`}>
                Market orders right now: {agrees ? 'AGREE with this trade' : against ? 'AGAINST this trade (be careful)' : 'no clear side'} ({flow.overall.label}).
              </p>
            )
          })()}
          <p className="plan-why">Why: {a.why}. Target is the next strong {long ? 'resistance above' : 'support below'};
            stop is beyond the next strong {long ? 'support below' : 'resistance above'}.</p>
        </div>
      ) : (
        <p className="muted">{plan?.message ? capitalise(plan.message) : 'Waiting for the first check.'}</p>
      )}

      {acct && (
        <dl className="plan-account">
          <Row label="Paper account">${fmtNum(acct.equity, 2)}</Row>
          <Row label="Closed trades">{acct.closed ? `${acct.wins} won of ${acct.closed}, ${acct.total_r >= 0 ? '+' : ''}${fmtNum(acct.total_r, 2)} R` : 'none yet'}</Row>
        </dl>
      )}

      {plan?.history?.length > 0 && (
        <>
          <h3>Recent plans</h3>
          <ul className="plan-history">
            {plan.history.slice(0, 5).map((h, i) => (
              <li key={i}>
                <span className={h.side === 'long' ? 'up' : 'down'}>{h.side === 'long' ? 'Buy' : 'Sell'} {fmtPrice(h.entry, symbol)}</span>
                <span className="muted">{h.status === 'closed'
                  ? `${h.result}, ${h.r >= 0 ? '+' : ''}${fmtNum(h.r, 2)} R`
                  : `cancelled: ${h.result}`}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="footnote">Paper orders only: nothing is sent to MEXC. One plan per coin at a time; a pending
        order is cancelled if the wall behind it is pulled.</p>
    </section>
  )
}

const capitalise = (s) => s.charAt(0).toUpperCase() + s.slice(1)
