import { useEffect, useImperativeHandle, useRef, forwardRef } from 'react'
import { CandlestickSeries, HistogramSeries, createChart } from 'lightweight-charts'
import { fmtPrice, pricePrecision } from '../api.js'
import { PositionBoxes } from './positionBoxes.js'

const C = {
  ground: '#FFFFFF',
  grid: '#F1F3F5',
  text: '#5F6B76',
  line: '#E1E5E9',
  up: '#089981',
  down: '#F23645',
  upVol: 'rgba(8, 153, 129, 0.35)',
  downVol: 'rgba(242, 54, 69, 0.35)',
  level: '#5B6CC9',
  tpFill: 'rgba(8, 153, 129, 0.16)',
  tpBorder: 'rgba(8, 153, 129, 0.55)',
  slFill: 'rgba(242, 54, 69, 0.16)',
  slBorder: 'rgba(242, 54, 69, 0.5)',
  entry: '#1C2630',
}

// Chart times are UTC seconds; show them in the viewer's local time zone.
const two = (n) => String(n).padStart(2, '0')
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
function tickFormatter(time, type) {
  const d = new Date(time * 1000)
  if (type === 0) return String(d.getFullYear())
  if (type === 1) return MONTHS[d.getMonth()]
  if (type === 2) return `${d.getDate()} ${MONTHS[d.getMonth()]}`
  return `${two(d.getHours())}:${two(d.getMinutes())}`
}
function crosshairTime(time) {
  const d = new Date(time * 1000)
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}  ${two(d.getHours())}:${two(d.getMinutes())}`
}

const volBar = (b) => ({ time: b.time, value: b.volume, color: b.close >= b.open ? C.upVol : C.downVol })

/** Candles + volume. Parent calls ref.setBars(bars) and ref.updateBars(bars). */
const PriceChart = forwardRef(function PriceChart({ symbol }, ref) {
  const el = useRef(null)
  const api = useRef({})

  useEffect(() => {
    const chart = createChart(el.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: C.ground },
        textColor: C.text,
        fontFamily: "'Archivo Variable', system-ui, sans-serif",
        fontSize: 11,
        attributionLogo: true,
        panes: { separatorColor: C.line, separatorHoverColor: C.line, enableResize: false },
      },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.line },
      timeScale: { borderColor: C.line, timeVisible: true, secondsVisible: false, rightOffset: 24,
                   tickMarkFormatter: tickFormatter },
      localization: { timeFormatter: crosshairTime },
      crosshair: { mode: 0 },
    })
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: C.up, downColor: C.down, wickUpColor: C.up, wickDownColor: C.down, borderVisible: false,
    })
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false,
    }, 1)
    chart.panes()[0].setStretchFactor(4)
    chart.panes()[1].setStretchFactor(1)
    const boxes = new PositionBoxes()
    candles.attachPrimitive(boxes)
    api.current = { chart, candles, volume, boxes, lastTime: 0, times: [] }
    return () => chart.remove()
  }, [])

  useEffect(() => {
    const p = pricePrecision(symbol)
    api.current.candles?.applyOptions({ priceFormat: { type: 'price', precision: p, minMove: 1 / 10 ** p } })
  }, [symbol])

  useImperativeHandle(ref, () => ({
    setBars(bars) {
      const { chart, candles, volume } = api.current
      candles.setData(bars)
      volume.setData(bars.map(volBar))
      api.current.lastTime = bars.length ? bars[bars.length - 1].time : 0
      api.current.times = bars.map((b) => b.time)
      api.current.boxes.set(api.current.boxList ?? [], api.current.lastTime)
      const n = bars.length
      // roughly 8 px per candle, so narrow screens show fewer candles
      const show = Math.max(40, Math.min(200, Math.round((el.current?.clientWidth ?? 1000) / 8)))
      if (n) chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - show), to: n + 24 })
    },
    /** 4-hour support / resistance: [{price, kind, touches}] */
    setLevels(levels, sym) {
      const { candles } = api.current
      if (!candles) return
      const sig = levels.map((l) => `${l.price}${l.touches}`).join('|')
      if (sig === api.current.levelSig) return
      api.current.levelSig = sig
      for (const l of api.current.levelLines ?? []) candles.removePriceLine(l)
      api.current.levelLines = levels.map((l) => candles.createPriceLine({
        price: l.price,
        color: C.level,
        lineWidth: l.score >= 6 ? 3 : 2,
        lineStyle: 0,
        axisLabelVisible: true,
        title: `${l.kind === 'resistance' ? 'R' : 'S'}${l.daily ? ' D' : ' 4h'} ×${l.touches}`,
      }))
    },
    /** Trade plan: entry (limit), stop, target as lines plus green / red boxes. */
    setPlan(plan, sym) {
      const { candles, boxes } = api.current
      if (!candles) return
      const sig = plan ? `${plan.status}${plan.entry}${plan.stop}${plan.target}` : ''
      if (sig === api.current.planSig) return
      api.current.planSig = sig
      for (const l of api.current.planLines ?? []) candles.removePriceLine(l)
      api.current.planLines = []
      api.current.boxList = []
      if (plan) {
        const long = plan.side === 'long'
        const pending = plan.status === 'pending'
        const t = Math.floor(new Date(plan.placed_at).getTime() / 1000)
        const times = api.current.times
        let bar = null
        for (let i = times.length - 1; i >= 0; i--) if (times[i] <= t) { bar = times[i]; break }
        api.current.boxList = [
          { time: bar, from: plan.entry, to: plan.target, fill: C.tpFill, border: C.tpBorder },
          { time: bar, from: plan.entry, to: plan.stop, fill: C.slFill, border: C.slBorder },
        ]
        const line = (price, color, title, style = 0) => candles.createPriceLine({
          price, color, lineWidth: 2, lineStyle: style, axisLabelVisible: true, title })
        api.current.planLines = [
          line(plan.entry, C.entry, pending ? `${long ? 'BUY' : 'SELL'} LIMIT` : `${long ? 'LONG' : 'SHORT'}`,
               pending ? 2 : 0),
          line(plan.target, C.up, 'TP'),
          line(plan.stop, C.down, 'SL'),
        ]
      }
      boxes.set(api.current.boxList, api.current.lastTime)
    },
    /** Dashed lines for large resting orders: [{side, price, size, label}] */
    setLargeOrders(orders) {
      const { candles } = api.current
      if (!candles) return
      const sig = orders.map((o) => `${o.side}${o.price}${o.label}`).join('|')
      if (sig === api.current.lineSig) return
      api.current.lineSig = sig
      for (const l of api.current.lines ?? []) candles.removePriceLine(l)
      api.current.lines = orders.map((o) => candles.createPriceLine({
        price: o.price,
        color: o.side === 'bid' ? C.up : C.down,
        lineWidth: 1,
        lineStyle: 2, // dashed
        axisLabelVisible: true,
        title: o.label,
      }))
    },
    updateBars(bars) {
      const { candles, volume } = api.current
      for (const b of bars) {
        if (b.time < api.current.lastTime) continue // older than what's drawn
        candles.update(b)
        volume.update(volBar(b))
        if (b.time > api.current.lastTime) api.current.times.push(b.time)
        api.current.lastTime = b.time
      }
    },
  }), [])

  return <div className="chart" ref={el} />
})

export default PriceChart
