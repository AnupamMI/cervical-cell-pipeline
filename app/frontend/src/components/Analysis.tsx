import { useEffect, useState } from 'react'
import type { Options } from '../lib/api'

export interface Stage {
  key: 'detect' | 'consensus' | 'clusters' | 'segment' | 'gate'
  label: string
  detail: string
}

export const STAGES: Stage[] = [
  { key: 'detect', label: 'Detect', detail: 'YOLO26s finds the cells' },
  { key: 'consensus', label: 'Consensus', detail: 'YOLO26l cross-checks them' },
  { key: 'clusters', label: 'Clusters & debris', detail: 'Detector C outlines clusters and debris' },
  { key: 'segment', label: 'Segment', detail: 'TinyUNet outlines cell and nucleus' },
  { key: 'gate', label: 'Quality gate', detail: 'Each mask is sorted by status' },
]
const FALLBACK = { detect: 400, consensus: 2500, clusters: 600, segment: 1200, gate: 500 }

/**
 * The API call is synchronous (one request, one response), so the current stage is an estimate from the warm-up
 * timings in /api/health; the measured stage times appear in the side panel afterwards.
 */
export function Analysis({ name, opts, typical, onCancel }: {
  name: string
  opts: Options
  typical: Record<string, number> | null
  onCancel: () => void
}) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t0 = performance.now()
    const id = window.setInterval(() => setElapsed(performance.now() - t0), 100)
    return () => clearInterval(id)
  }, [])
  const stages = STAGES.filter((s) => (s.key !== 'consensus' || opts.consensus) && (s.key !== 'clusters' || opts.clusters))
  const est = stages.map((s) => typical?.[s.key] || FALLBACK[s.key])
  const pre = 400                   // upload / decode
  const total = pre + est.reduce((a, b) => a + b, 0)
  let acc = pre, current = stages.length - 1
  for (let i = 0; i < stages.length; i++) {
    acc += est[i]
    if (elapsed < acc) { current = i; break }
  }
  const stage = elapsed < pre ? null : stages[current]
  const frac = Math.min(0.95, elapsed / total)

  return (
    <main id="main" className="analysis">
      <section className="card analysis__card" aria-labelledby="an-h">
        <p className="eyebrow">Analysing</p>
        <h1 id="an-h" className="analysis__title">{name}</h1>
        <div className="progress" role="progressbar" aria-label="Estimated progress" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={Math.round(frac * 100)}>
          <span style={{ transform: `scaleX(${frac})` }} />
        </div>
        <ol className="steps" aria-label="Pipeline stages">
          {stages.map((s, i) => {
            const state = stage === null ? 'todo' : i < current ? 'done' : i === current ? 'now' : 'todo'
            return (
              <li key={s.key} className={`step step--${state}`} aria-current={state === 'now' ? 'step' : undefined}>
                <span className="step__mark" aria-hidden="true">
                  {state === 'done' && <svg viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" /></svg>}
                </span>
                <span className="step__label">{s.label}</span>
                <span className="step__detail">{s.detail}</span>
              </li>
            )
          })}
        </ol>
        <p className="sr-only" aria-live="polite">{stage ? `${stage.label}: ${stage.detail}` : 'Uploading'}</p>
        <div className="analysis__foot">
          <span className="muted small">{(elapsed / 1000).toFixed(1)} s{stage ? '' : ' · uploading'}</span>
          <button type="button" className="btn btn--quiet" onClick={onCancel}>Cancel</button>
        </div>
      </section>
    </main>
  )
}
