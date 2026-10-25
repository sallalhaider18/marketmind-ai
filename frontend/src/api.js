import { useEffect, useRef, useState } from 'react'

export async function getJSON(path) {
  const r = await fetch(path)
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${detail}`)
  }
  return r.json()
}

/**
 * Subscribe to /ws/live/{symbol}. Returns { message, backendOnline }.
 * `onMessage` is called for every push (kept in a ref so it can change freely).
 * Reconnects every 2 s if the backend goes away.
 */
export function useLiveFeed(symbol, interval, onMessage) {
  const [message, setMessage] = useState(null)
  const [backendOnline, setBackendOnline] = useState(true)
  const cb = useRef(onMessage)
  cb.current = onMessage

  useEffect(() => {
    let ws
    let retry
    let closed = false
    setMessage(null)

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws/live/${symbol}?interval=${interval}`)
      ws.onopen = () => setBackendOnline(true)
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        setMessage(msg)
        cb.current?.(msg)
      }
      ws.onclose = () => {
        if (closed) return
        setBackendOnline(false)
        retry = setTimeout(connect, 2000)
      }
    }
    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      ws?.close()
    }
  }, [symbol, interval])

  return { message, backendOnline }
}

// Display precision per instrument (matches MEXC tick size)
const PRECISION = { 'BTCUSDT.P': 1, 'ETHUSDT.P': 2 }
export const pricePrecision = (symbol) => PRECISION[symbol] ?? 2
export const baseAsset = (symbol) => symbol.replace(/USDT(\.P)?$/, '')

export const fmtPrice = (v, symbol) =>
  v == null ? '—' : v.toLocaleString('en-GB', {
    minimumFractionDigits: pricePrecision(symbol),
    maximumFractionDigits: pricePrecision(symbol),
  })

export const fmtNum = (v, digits = 2) =>
  v == null ? '—' : v.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const fmtTime = (iso) =>
  iso ? new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'
