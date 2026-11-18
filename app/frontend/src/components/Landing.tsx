import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { ACCEPT, checkFile, type Options, type Sample } from '../lib/api'
import { Switch } from './Switch'

type Glass = typeof import('./BubbleSwitch')

// Approved Light glass material (catalog defaults for light pages).
const GLASS = {
  glass: 0.95, iridescence: 0.5, filmScale: 4.85, oilOpacity: 1, oilCoverage: 0.73, oilEdge: 18,
  oilEdgeFade: 2.33, clearCenter: 0.57, thickness: 480, drainage: 0.35, swirl: 0.5,
  reflectivity: 1.3, refraction: 0.15, dispersion: 1, bevel: 14,
  windowLength: 1.3, windowMargin: 0.85, windowBend: 0.75, windowSize: 0.75, windowWidth: 0.8,
  windowGradient: 0.96, windowHead: 0.7, windowTail: 0.6, glintSize: 1.65, glintIntensity: 0,
  glintDistance: -0.45, glintHeight: 0.65, windowIntensity: 0.85, windowTwin: false,
  bottomShadeColor: '#ffffff',
} as const

// Shares the glass material and the card backdrop once the bubble module has loaded.
function Mat({ glass, card, children }: { glass: Glass | null; card: RefObject<HTMLElement | null>; children: ReactNode }) {
  return glass ? <glass.SoapBubbleMaterial backdrop={card} {...GLASS}>{children}</glass.SoapBubbleMaterial> : <>{children}</>
}

const STEPS: { n: string; t: string; d: string; opt?: keyof Options }[] = [
  { n: 'YOLO26s', t: 'Find', d: 'Proposes every cell' },
  { n: 'Detector C', t: 'Confirm', d: 'Agrees it is a cell; marks clusters and debris', opt: 'clusters' },
  { n: 'YOLO26l', t: 'Confirm', d: 'Agrees too, at higher resolution', opt: 'consensus' },
  { n: 'TinyUNet', t: 'Segment', d: 'Outlines cell and nucleus' },
  { n: 'Quality gate', t: 'Sort', d: 'Measurable, review or excluded' },
]

export function Landing({ samples, samplesError, onFile, onSample, opts, avail, setOpt, error, busyRetry }: {
  samples: Sample[] | null
  samplesError: boolean
  onFile: (f: File) => void
  onSample: (name: string) => void
  opts: Options
  avail: Options
  setOpt: (k: keyof Options, v: boolean) => void
  error: string | null
  busyRetry: (() => void) | null
}) {
  const input = useRef<HTMLInputElement>(null)
  const card = useRef<HTMLElement>(null)
  // Plain controls first; on mouse/trackpad devices the bubble-glass versions load once the page is idle.
  const [glass, setGlass] = useState<Glass | null>(null)
  useEffect(() => {
    // Decorative only: skip on touch-first devices, where the setup cost is felt as lag.
    if (!matchMedia('(hover: hover) and (pointer: fine)').matches) return
    let live = true
    const load = () => import('./BubbleSwitch').then((m) => live && setGlass(m), () => {})
    const w = window as Window & { requestIdleCallback?: (f: () => void, o?: { timeout: number }) => number }
    const id = w.requestIdleCallback ? w.requestIdleCallback(load, { timeout: 2500 }) : window.setTimeout(load, 1200)
    return () => { live = false; if (!w.requestIdleCallback) clearTimeout(id) }
  }, [])
  const Sw = glass?.BubbleSwitch ?? Switch
  const choose = () => input.current?.click()
  const [drag, setDrag] = useState(false)
  const [local, setLocal] = useState<string | null>(null)

  const take = (f: File | undefined) => {
    if (!f) return
    const e = checkFile(f)
    setLocal(e)
    if (!e) onFile(f)
  }
  const msg = local ?? error
  const on = (k?: keyof Options) => !k || (opts[k] && avail[k])

  return (
    <main id="main" className="landing">
      <div className="hero">
        <div className="landing__intro">
          <p className="eyebrow eyebrow--accent">Pap-smear field of view analysis</p>
          <h1 className="landing__title">Find, confirm and outline every cell.</h1>
          <p className="landing__lede">
            Three detectors must agree before a cell is segmented. Clusters and debris are marked, never measured.
          </p>
          <ol className="flow" aria-label="Pipeline">
            {STEPS.map((s, i) => (
              <li key={s.n} className={`flow__step ${on(s.opt) ? '' : 'is-off'}`}>
                <span className="flow__n" aria-hidden="true">{i + 1}</span>
                <span className="flow__body">
                  <span className="flow__t">{s.t} <span className="flow__model">· {s.n}</span></span>
                  <span className="flow__d">{s.d}{on(s.opt) ? '' : ' (off)'}</span>
                </span>
              </li>
            ))}
          </ol>
        </div>

        <section ref={card} className="card upload" aria-labelledby="upload-h">
          <Mat glass={glass} card={card}>
          <div className={`drop ${drag ? 'is-drag' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files[0]) }}>
            <span className="drop__badge" aria-hidden="true">
              <svg className="drop__icon" viewBox="0 0 24 24">
                <path d="M12 15V4m0 0L8 8m4-4 4 4M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
              </svg>
            </span>
            <h2 id="upload-h" className="drop__h">Drop an image here</h2>
            <p className="drop__meta">TIFF, PNG or JPEG · 2028 × 1520 · up to 25 MB</p>
            {glass
              ? <glass.SoapBubbleButton className="bbtn" tint="#3d35a8" opacity={0.92} onClick={choose}>Choose image</glass.SoapBubbleButton>
              : <button type="button" className="btn btn--primary bbtn-plain" onClick={choose}>Choose image</button>}
            <input ref={input} type="file" accept={ACCEPT} hidden aria-hidden="true" tabIndex={-1}
              onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
          </div>
          <div className="upload__opts" role="group" aria-label="Pipeline settings">
            <Sw checked={opts.vote} onChange={(v) => setOpt('vote', v)} label="Segment confirmed cells only"
              hint="Off: every YOLO26s cell is segmented." />
            <Sw checked={opts.clusters && avail.clusters} disabled={!avail.clusters}
              onChange={(v) => setOpt('clusters', v)} label="Detector C"
              hint={avail.clusters ? 'Cell check, clusters and debris.' : 'Not available here.'} />
            <Sw checked={opts.consensus && avail.consensus} disabled={!avail.consensus}
              onChange={(v) => setOpt('consensus', v)} label="YOLO26l"
              hint={avail.consensus ? 'Second cell check.' : 'Not available here.'} />
          </div>
          </Mat>
          {msg && <p className="error" role="alert">{msg}</p>}
          {busyRetry && (
            <p className="error error--busy" role="status">
              <span>Another image is being analysed.</span>
              <button type="button" className="btn btn--quiet btn--sm" onClick={busyRetry}>Retry</button>
            </p>
          )}
        </section>
      </div>

      <section className="samples" aria-labelledby="samples-h">
        <h2 id="samples-h" className="section-h">Or try a sample</h2>
        {samplesError && <p className="muted">Samples could not be loaded.</p>}
        <div className="samples__row">
          {samples === null && !samplesError &&
            Array.from({ length: 5 }, (_, i) => <span key={i} className="sample sample--skeleton" aria-hidden="true" />)}
          {samples?.slice(0, 5).map((s, i) => (
            <button key={s.name} type="button" className="sample" onClick={() => onSample(s.name)}
              aria-label={`Analyse sample ${i + 1} (${s.name})`}>
              <img src={s.thumb} alt="" width={160} height={120} decoding="async" />
              <span>Sample {i + 1}</span>
            </button>
          ))}
        </div>
      </section>

      <p className="landing__foot muted small">Research demo, not for diagnosis. Images are deleted after decoding; results after 15 minutes.</p>
    </main>
  )
}
