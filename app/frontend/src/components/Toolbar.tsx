import type { Options } from '../lib/api'
import type { Layers } from './Viewer'
import { Switch } from './Switch'

const LAYERS: { key: keyof Layers; label: string }[] = [
  { key: 'detections', label: 'Detections' },
  { key: 'masks', label: 'Masks' },
  { key: 'status', label: 'Status' },
  { key: 'clusters', label: 'Clusters & debris' },
  { key: 'flaggedOnly', label: 'Flagged only' },
]

export function Toolbar({ layers, setLayers, opts, avail, setOpt, clustersShown, busy, onZoom, onHome }: {
  layers: Layers
  setLayers: (l: Layers) => void
  opts: Options
  avail: Options
  setOpt: (k: keyof Options, v: boolean) => void
  clustersShown: boolean
  busy: boolean
  onZoom: (f: number) => void
  onHome: () => void
}) {
  return (
    <div className="toolbar">
      <div className="segmented" role="group" aria-label="Layers">
        {LAYERS.map((t) => {
          const off = t.key === 'clusters' && !clustersShown
          return (
            <button key={t.key} type="button" className="segmented__btn" aria-pressed={layers[t.key] && !off}
              disabled={off} title={off ? 'Turn on “Clusters” to run detector C' : undefined}
              onClick={() => setLayers({ ...layers, [t.key]: !layers[t.key] })}>
              {t.label}
            </button>
          )
        })}
      </div>
      <div className="toolbar__right">
        <Switch compact checked={opts.consensus && avail.consensus} disabled={!avail.consensus || busy}
          onChange={(v) => setOpt('consensus', v)} label="Consensus" />
        <Switch compact checked={opts.clusters && avail.clusters} disabled={!avail.clusters || busy}
          onChange={(v) => setOpt('clusters', v)} label="Clusters" />
        <Switch compact checked={opts.vote} disabled={busy} onChange={(v) => setOpt('vote', v)} label="Confirm" />
        <div className="zoom" role="group" aria-label="Zoom">
          <button type="button" className="iconbtn" onClick={() => onZoom(1 / 1.4)} aria-label="Zoom out">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9" /></svg>
          </button>
          <button type="button" className="iconbtn" onClick={() => onZoom(1.4)} aria-label="Zoom in">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9M8 3.5v9" /></svg>
          </button>
          <button type="button" className="iconbtn" onClick={onHome} aria-label="Fit image">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" /></svg>
          </button>
        </div>
      </div>
    </div>
  )
}
