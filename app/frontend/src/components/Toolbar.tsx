import type { Layers } from './Viewer'
import { Glass } from './Glass'
import { Switch } from './Switch'

const TOGGLES: { key: keyof Layers; label: string }[] = [
  { key: 'detections', label: 'Detections' },
  { key: 'masks', label: 'Masks' },
  { key: 'status', label: 'Status colours' },
  { key: 'flaggedOnly', label: 'Flagged only' },
]

export function Toolbar({ layers, setLayers, consensus, setConsensus, consensusAvailable, busy, onZoom, onHome }: {
  layers: Layers
  setLayers: (l: Layers) => void
  consensus: boolean
  setConsensus: (v: boolean) => void
  consensusAvailable: boolean
  busy: boolean
  onZoom: (f: number) => void
  onHome: () => void
}) {
  return (
    <Glass as="div" radius={999} className="toolbar" role="toolbar" aria-label="Viewer layers">
      <div className="toolbar__group" role="group" aria-label="Layers">
        {TOGGLES.map((t) => (
          <button key={t.key} type="button" className="tbtn" aria-pressed={layers[t.key]}
            onClick={() => setLayers({ ...layers, [t.key]: !layers[t.key] })}>
            <span className="tbtn__box" aria-hidden="true" />{t.label}
          </button>
        ))}
      </div>
      <span className="toolbar__sep" aria-hidden="true" />
      <Switch compact checked={consensus && consensusAvailable} disabled={!consensusAvailable || busy}
        onChange={setConsensus} label={busy ? 'Re-running…' : 'Consensus check (YOLO26l)'} />
      <span className="toolbar__sep" aria-hidden="true" />
      <div className="toolbar__group" role="group" aria-label="Zoom">
        <button type="button" className="tbtn tbtn--icon" onClick={() => onZoom(1 / 1.4)} aria-label="Zoom out">−</button>
        <button type="button" className="tbtn tbtn--icon" onClick={() => onZoom(1.4)} aria-label="Zoom in">+</button>
        <button type="button" className="tbtn tbtn--icon" onClick={onHome} aria-label="Fit image">⤢</button>
      </div>
    </Glass>
  )
}
