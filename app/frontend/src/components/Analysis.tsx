import { useEffect, useState } from 'react'
import { Glass } from './Glass'

export interface Stage {
  key: 'detect' | 'consensus' | 'segment' | 'gate'
  label: string
  model: string
}

export const STAGES: Stage[] = [
  { key: 'detect', label: 'Detect', model: 'YOLO26s finds cell boxes' },
  { key: 'consensus', label: 'Consensus', model: 'YOLO26l @1280 cross-checks them' },
  { key: 'segment', label: 'Segment', model: 'TinyUNet outlines cell + nucleus' },
  { key: 'gate', label: 'Gate', model: 'Quality gate sorts every mask' },
]

/**
 * The API call is synchronous (one request, one response), so the stage shown here is an estimate from the
 * warm-up timings in /api/health (scaled by elapsed time); the real per-stage times appear in the side panel.
 */
export function Analysis({ name, consensus, typical, onCancel }: {
  name: string
  consensus: boolean
  typical: Record<string, number> | null
  onCancel: () => void
}) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t0 = performance.now()
    const id = window.setInterval(() => setElapsed(performance.now() - t0), 100)
    return () => clearInterval(id)
  }, [])
  const stages = STAGES.filter((s) => consensus || s.key !== 'consensus')
  const est = stages.map((s) => typical?.[s.key] ?? { detect: 400, consensus: 2500, segment: 1200, gate: 500 }[s.key])
  // upload / decode before the models start
  const pre = 400
  let acc = pre, current = stages.length - 1
  for (let i = 0; i < stages.length; i++) {
    acc += est[i]
    if (elapsed < acc) { current = i; break }
  }
  const stage = elapsed < pre ? null : stages[current]

  return (
    <main id="main" className="analysis">
      <Glass variant="text" radius={26} className="analysis__card">
        <p className="eyebrow">Analysing</p>
        <h1 className="analysis__title">{name}</h1>
        <ol className="stages" aria-label="Pipeline stages">
          {stages.map((s, i) => {
            const state = stage === null ? 'todo' : i < current ? 'done' : i === current ? 'now' : 'todo'
            return (
              <li key={s.key} className={`stage stage--${state}`} aria-current={state === 'now' ? 'step' : undefined}>
                <span className="stage__dot" aria-hidden="true" />
                <span className="stage__label">{s.label}</span>
                <span className="stage__model">{s.model}</span>
              </li>
            )
          })}
        </ol>
        <p className="sr-only" aria-live="polite">{stage ? `${stage.label}: ${stage.model}` : 'Uploading'}</p>
        <div className="analysis__foot">
          <span className="muted">{(elapsed / 1000).toFixed(1)} s{stage ? '' : ' · uploading'}</span>
          <button type="button" className="btn btn--ghost" onClick={onCancel}>Cancel</button>
        </div>
      </Glass>
    </main>
  )
}
