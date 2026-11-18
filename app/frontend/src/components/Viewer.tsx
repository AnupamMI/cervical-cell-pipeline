import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react'
import OpenSeadragon from 'openseadragon'
import type { Cell, Pt, Result } from '../lib/api'

export interface Layers {
  detections: boolean
  masks: boolean
  status: boolean
  clusters: boolean
  skipped: boolean
  flaggedOnly: boolean
}

export interface ViewerHandle {
  zoomBy(f: number): void
  home(): void
  zoomToCell(id: number): void
}

const pathOf = (polys: Pt[][]) =>
  polys.map((p) => 'M' + p.map(([x, y]) => `${x},${y}`).join('L') + 'Z').join('')

/**
 * OpenSeadragon (BSD-3) zoom/pan on the FOV (display JPEG from the API) with an SVG overlay in image pixels.
 * The overlay's transform follows the viewport on every OSD frame (direct DOM write, no React re-render); strokes
 * use vector-effect: non-scaling-stroke so they stay 1-2 px at any zoom. Clicks are hit-tested in image coordinates
 * (smallest box containing the point), so OSD keeps drag / pinch / keyboard navigation.
 */
const Viewer = forwardRef<ViewerHandle, {
  result: Result
  layers: Layers
  selected: number | null
  onSelect: (id: number | null) => void
  reducedMotion: boolean
  phone: boolean
}>(function Viewer({ result, layers, selected, onSelect, reducedMotion, phone }, ref) {
  const host = useRef<HTMLDivElement>(null)
  const g = useRef<SVGGElement>(null)
  const viewer = useRef<OpenSeadragon.Viewer | null>(null)
  const cellsRef = useRef<Cell[]>(result.cells)
  const selectRef = useRef(onSelect)
  cellsRef.current = result.cells
  selectRef.current = onSelect
  const { width: W, height: H } = result.size

  useEffect(() => {
    if (!host.current) return
    const v = OpenSeadragon({
      element: host.current,
      tileSources: { type: 'image', url: result.downloads['image.jpg'], buildPyramid: true } as unknown as string,
      showNavigationControl: false,
      showNavigator: !phone,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorSizeRatio: 0.16,
      navigatorAutoFade: true,
      animationTime: reducedMotion ? 0 : 0.7,
      springStiffness: reducedMotion ? 100 : 8,
      maxZoomPixelRatio: 4,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.6,
      constrainDuringPan: true,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true },
      gestureSettingsTouch: { pinchRotate: false, dblClickToZoom: true },
      crossOriginPolicy: false,
      preserveImageSizeOnResize: true,
    })
    viewer.current = v
    const sync = () => {
      const item = v.world.getItemAt(0)
      if (!item || !g.current) return
      const p0 = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0))
      const p1 = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(W, 0))
      const k = (p1.x - p0.x) / W
      g.current.setAttribute('transform', `translate(${p0.x} ${p0.y}) scale(${k})`)
      // labels keep a constant on-screen size: font-size in image units = px / scale
      g.current.style.setProperty('--k', String(1 / k))
      g.current.classList.toggle('is-near', k > 0.7)   // detail labels only once zoomed in
    }
    v.addHandler('open', () => {
      sync()
      host.current?.classList.add('is-open')
    })
    v.addHandler('animation', sync)
    v.addHandler('update-viewport', sync)
    v.addHandler('resize', sync)
    v.addHandler('canvas-click', (e) => {
      if (!e.quick) return
      const item = v.world.getItemAt(0)
      if (!item) return
      const p = item.viewerElementToImageCoordinates(e.position!)
      const hit = cellsRef.current
        .filter((c) => p.x >= c.box[0] && p.x <= c.box[2] && p.y >= c.box[1] && p.y <= c.box[3])
        .sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))[0]
      selectRef.current(hit ? hit.id : null)
    })
    return () => {
      v.destroy()
      viewer.current = null
    }
    // the viewer is rebuilt only for a new image
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.id])

  useImperativeHandle(ref, () => ({
    zoomBy: (f) => {
      const v = viewer.current
      if (v) v.viewport.zoomBy(f).applyConstraints()
    },
    home: () => viewer.current?.viewport.goHome(),
    zoomToCell: (id) => {
      const v = viewer.current
      const c = cellsRef.current.find((x) => x.id === id)
      const item = v?.world.getItemAt(0)
      if (!v || !c || !item) return
      const [x1, y1, x2, y2] = c.box
      const pad = Math.max(x2 - x1, y2 - y1) * 0.6
      v.viewport.fitBounds(item.imageToViewportRectangle(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad), reducedMotion)
    },
  }), [reducedMotion])

  const shown = useMemo(
    () => result.cells.filter((c) => c.status !== 'not_segmented' && (!layers.flaggedOnly || c.status !== 'measurable')),
    [result.cells, layers.flaggedOnly],
  )
  const skipped = useMemo(() => result.cells.filter((c) => c.status === 'not_segmented'), [result.cells])
  const paths = useMemo(
    () => new Map(result.cells.map((c) => [c.id, { cell: pathOf(c.cell_polygons), nuc: pathOf(c.nucleus_polygons) }])),
    [result.cells],
  )
  const tone = (c: Cell) => (layers.status ? `st-${c.status}` : 'st-neutral')
  const sel = result.cells.find((c) => c.id === selected)

  return (
    <div className="viewer">
      <div ref={host} className="viewer__osd" role="application"
        aria-label={`Field of view, ${W} by ${H} pixels, ${result.cells.length} cells${result.clusters.used ? `, ${result.clusters.boxes.length} cluster or debris regions` : ''}. Drag to pan, scroll or +/- to zoom.`} />
      <svg className="viewer__overlay" aria-hidden="true">
        <g ref={g}>
          {layers.masks && (
            <g className="lyr-masks">
              {shown.map((c) => (
                <g key={c.id} className={tone(c)}>
                  {c.nucleus_polygons.length > 0 && <path className="nuc" d={paths.get(c.id)!.nuc} />}
                  {c.cell_polygons.length > 0 && <path className="cellline" d={paths.get(c.id)!.cell} />}
                  {c.cell_polygons.length === 0 && !layers.detections && (
                    <rect className="det" x={c.box[0]} y={c.box[1]} width={c.box[2] - c.box[0]} height={c.box[3] - c.box[1]} />
                  )}
                </g>
              ))}
            </g>
          )}
          {layers.detections && (
            <g className="lyr-dets">
              {shown.map((c) => (
                <rect key={c.id} className={`det ${tone(c)} ${c.det_tag === 's_only' ? 'is-sonly' : ''}`}
                  x={c.box[0]} y={c.box[1]} width={c.box[2] - c.box[0]} height={c.box[3] - c.box[1]} />
              ))}
            </g>
          )}
          {layers.skipped && (
            <g className="lyr-skip">
              {skipped.map((c) => (
                <rect key={c.id} className="det st-not_segmented"
                  x={c.box[0]} y={c.box[1]} width={c.box[2] - c.box[0]} height={c.box[3] - c.box[1]} rx={4} />
              ))}
            </g>
          )}
          {layers.clusters && result.clusters.used && (
            <g className="lyr-c">
              {result.clusters.boxes.map((b) => (
                <g key={b.id} className={`cbox cbox--${b.class}`}>
                  <rect x={b.box[0]} y={b.box[1]} width={b.box[2] - b.box[0]} height={b.box[3] - b.box[1]} rx={6} />
                  <text x={b.box[0]} y={b.box[1]} dx="0.4em" dy="1.25em">{b.class === 'cluster' ? 'Cluster' : 'Debris'} {b.conf.toFixed(2)}</text>
                </g>
              ))}
            </g>
          )}
          {sel && (
            <g className="lyr-sel">
              <rect className="sel-halo" x={sel.box[0] - 6} y={sel.box[1] - 6}
                width={sel.box[2] - sel.box[0] + 12} height={sel.box[3] - sel.box[1] + 12} rx={10} />
              {sel.cell_polygons.length > 0 && <path className="sel-line" d={paths.get(sel.id)!.cell} />}
            </g>
          )}
        </g>
      </svg>
    </div>
  )
})

export default Viewer
