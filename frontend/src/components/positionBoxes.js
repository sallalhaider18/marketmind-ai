// TradingView-style position boxes for lightweight-charts (series primitive):
// a green box from entry to target and a light-red box from entry to stop,
// starting at the time the plan was made and running to the right edge.

class BoxRenderer {
  constructor(src) { this.src = src }

  draw(target) {
    const { chart, series, boxes } = this.src
    if (!chart || !series || !boxes.length) return
    target.useBitmapCoordinateSpace(({ context: ctx, bitmapSize, horizontalPixelRatio: hr, verticalPixelRatio: vr }) => {
      const ts = chart.timeScale()
      for (const b of boxes) {
        const yA = series.priceToCoordinate(b.from)
        const yB = series.priceToCoordinate(b.to)
        if (yA == null || yB == null) continue
        let x = b.time != null ? ts.timeToCoordinate(b.time) : null
        if (x == null) x = b.time != null && b.time > (this.src.lastTime ?? 0) ? null : 0
        if (x == null) continue
        const left = Math.max(0, Math.round(x * hr))
        const top = Math.round(Math.min(yA, yB) * vr)
        const height = Math.max(1, Math.round(Math.abs(yA - yB) * vr))
        ctx.fillStyle = b.fill
        ctx.fillRect(left, top, bitmapSize.width - left, height)
        ctx.strokeStyle = b.border
        ctx.lineWidth = Math.max(1, Math.round(hr))
        ctx.strokeRect(left + 0.5, top + 0.5, bitmapSize.width - left - 1, height - 1)
      }
    })
  }
}

class BoxPaneView {
  constructor(src) { this.src = src }
  zOrder() { return 'bottom' }
  renderer() { return new BoxRenderer(this.src) }
}

export class PositionBoxes {
  constructor() {
    this.boxes = []
    this.views = [new BoxPaneView(this)]
  }

  attached({ chart, series, requestUpdate }) {
    this.chart = chart
    this.series = series
    this.requestUpdate = requestUpdate
  }

  detached() { this.chart = this.series = null }
  updateAllViews() {}
  paneViews() { return this.views }

  /** boxes: [{time (bar time, s), from, to, fill, border}] */
  set(boxes, lastTime) {
    this.boxes = boxes
    this.lastTime = lastTime
    this.requestUpdate?.()
  }
}
