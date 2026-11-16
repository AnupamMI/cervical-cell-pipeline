import { Component, lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { ApiError, getHealth, getSamples, infer, type Health, type Result, type Sample, type Source } from './lib/api'
import { useAfter, useFirstInteraction, usePageVisible, usePhone, useReducedMotion } from './lib/hooks'
import { Glass } from './components/Glass'
import { Landing } from './components/Landing'
import { Analysis } from './components/Analysis'
import type { Layers, ViewerHandle } from './components/Viewer'
import { Toolbar } from './components/Toolbar'
import { SidePanel } from './components/SidePanel'
import { StaticCell } from './components/StaticCell'

// Decorative WebGL, lazy-loaded after first paint, landing only (see README > Frontend > Effects)
const GradientBg = lazy(() => import('./effects/GradientBg'))
const CellScene = lazy(() => import('./effects/CellScene'))
// OpenSeadragon is only needed once a result exists: its own chunk, prefetched when an analysis starts
const loadViewer = () => import('./components/Viewer')
const Viewer = lazy(loadViewer)

class Quiet extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? null : this.props.children
  }
}

function hasWebGL() {
  try {
    const c = document.createElement('canvas')
    return !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

type Screen = 'landing' | 'analysing' | 'viewer'

export default function App() {
  const reduced = useReducedMotion()
  const phone = usePhone()
  const visible = usePageVisible()
  // First paint = static still frame of the gradient + static cell. The live effects start on the first
  // interaction; the 3D cell a moment after the gradient so their start-up work never shares one task.
  // Phones (lighter effects): moving gradient at reduced pixel density, the cell stays the static illustration.
  const interacted = useFirstInteraction()
  const cellReady = useAfter(interacted, 700)
  const [webgl] = useState(hasWebGL)

  const [screen, setScreen] = useState<Screen>('landing')
  const [health, setHealth] = useState<Health | null>(null)
  const [samples, setSamples] = useState<Sample[] | null>(null)
  const [samplesError, setSamplesError] = useState(false)
  const [consensus, setConsensus] = useState(true)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busyRetry, setBusyRetry] = useState<Source | null>(null)
  const [rerunning, setRerunning] = useState(false)
  const [srcName, setSrcName] = useState('')
  const [layers, setLayers] = useState<Layers>({ detections: true, masks: true, status: true, flaggedOnly: false })
  const [selected, setSelected] = useState<number | null>(null)
  const source = useRef<Source | null>(null)
  const abort = useRef<AbortController | null>(null)
  const viewer = useRef<ViewerHandle>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    getSamples().then(setSamples).catch(() => setSamplesError(true))
  }, [])

  const consensusAvailable = health?.consensus_available ?? true

  const run = useCallback(async (src: Source, cons: boolean, inPlace = false) => {
    abort.current?.abort()
    const ac = new AbortController()
    abort.current = ac
    source.current = src
    setError(null)
    setBusyRetry(null)
    loadViewer()
    if (inPlace) setRerunning(true)
    else {
      setSrcName(src.kind === 'file' ? src.file.name : `Sample ${src.name}`)
      setScreen('analysing')
    }
    try {
      const r = await infer(src, cons, ac.signal)
      setResult(r)
      setSelected(null)
      setScreen('viewer')
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      const msg = e instanceof ApiError ? e.message : 'Could not reach the server. Check your connection and try again.'
      if (e instanceof ApiError && e.code === 'busy') setBusyRetry(src)
      setError(msg)
      if (!inPlace) setScreen('landing')
    } finally {
      if (abort.current === ac) setRerunning(false)
    }
  }, [])

  const cancel = () => {
    abort.current?.abort()
    setScreen(result ? 'viewer' : 'landing')
  }
  const goNew = () => {
    setScreen('landing')
    setResult(null)
    setSelected(null)
    setError(null)
  }
  const toggleConsensus = (v: boolean) => {
    setConsensus(v)
    if (screen === 'viewer' && source.current) run(source.current, v, true)
  }
  const select = (id: number | null, zoom = false) => {
    setSelected(id)
    if (zoom && id != null) viewer.current?.zoomToCell(id)
  }

  const effects = !reduced && webgl && interacted
  const typical = health?.typical_ms ? health.typical_ms[consensus ? 'consensus' : 'no_consensus'] : null

  return (
    <div className={`app app--${screen}`}>
      <a className="skip" href="#main">Skip to content</a>
      <div className="bg" aria-hidden="true" />
      <div className="fx-gradient fx-gradient--still is-ready" style={{ backgroundImage: 'url(/gradient-still.webp)' }}
        aria-hidden="true" />
      {effects && (
        <Quiet>
          <Suspense fallback={null}>
            <GradientBg frozen={screen !== 'landing'} lite={phone} active={visible} />
          </Suspense>
        </Quiet>
      )}
      <div className="bg-veil" aria-hidden="true" />

      <Glass as="header" variant="text" radius={18} className="header">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); goNew() }}>
          <svg viewBox="0 0 32 32" className="brand__mark" aria-hidden="true">
            <ellipse cx="16" cy="16" rx="13" ry="11" fill="#F6C1D9" fillOpacity=".45" />
            <circle cx="18" cy="15" r="4.2" fill="#4B3FB3" />
          </svg>
          <span className="brand__name">Cervical Cell Pipeline</span>
        </a>
        <span className="pill"><span className="pill__long">Research demo · </span>not for diagnosis</span>
      </Glass>

      {screen === 'landing' && (
        <Landing samples={samples} samplesError={samplesError} consensus={consensus} setConsensus={setConsensus}
          consensusAvailable={consensusAvailable}
          error={error}
          onFile={(f) => run({ kind: 'file', file: f }, consensus)}
          onSample={(n) => run({ kind: 'sample', name: n }, consensus)}
          cellSlot={
            <div className="fx-cell">
              <StaticCell />
              {effects && cellReady && !phone && (
                <Quiet>
                  <Suspense fallback={null}>
                    <CellScene active={visible} lite={phone} />
                  </Suspense>
                </Quiet>
              )}
            </div>
          } />
      )}
      {screen === 'landing' && busyRetry && (
        <div className="toast" role="status">
          <span>The demo is busy with another image.</span>
          <button type="button" className="btn btn--sm btn--primary" onClick={() => run(busyRetry, consensus)}>Retry</button>
        </div>
      )}

      {screen === 'analysing' && (
        <Analysis name={srcName} consensus={consensus && consensusAvailable} typical={typical} onCancel={cancel} />
      )}

      {screen === 'viewer' && result && (
        <main id="main" className="workspace">
          <div className="workspace__main">
            <Toolbar layers={layers} setLayers={setLayers} consensus={consensus} setConsensus={toggleConsensus}
              consensusAvailable={consensusAvailable} busy={rerunning}
              onZoom={(f) => viewer.current?.zoomBy(f)} onHome={() => viewer.current?.home()} />
            <div className="legend" aria-label="Legend">
              <span className="lg lg--measurable">Measurable</span>
              <span className="lg lg--review">Review</span>
              <span className="lg lg--excluded">Excluded</span>
              <span className="lg lg--solid">Confirmed</span>
              <span className="lg lg--dashed">YOLO26s only</span>
            </div>
            {error && <p className="inline-error" role="alert">{error}</p>}
            <div className={`viewer-frame ${rerunning ? 'is-busy' : ''}`}>
              <Suspense fallback={<div className="viewer-busy" role="status">Loading viewer…</div>}>
                <Viewer ref={viewer} result={result} layers={layers} selected={selected} onSelect={(id) => select(id)}
                  reducedMotion={reduced} phone={phone} />
              </Suspense>
              {rerunning && <div className="viewer-busy" role="status">Re-running the pipeline…</div>}
            </div>
          </div>
          <SidePanel result={result} selected={selected} onSelect={select} onNew={goNew} />
        </main>
      )}
    </div>
  )
}
