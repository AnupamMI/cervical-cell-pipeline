import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, getHealth, getSamples, infer, type Health, type Options, type Result, type Sample, type Source } from './lib/api'
import { usePhone, useReducedMotion } from './lib/hooks'
import { Landing } from './components/Landing'
import { Analysis } from './components/Analysis'
import type { Layers, ViewerHandle } from './components/Viewer'
import { Toolbar } from './components/Toolbar'
import { SidePanel } from './components/SidePanel'

// OpenSeadragon is only needed once a result exists: its own chunk, prefetched when an analysis starts
const loadViewer = () => import('./components/Viewer')
const Viewer = lazy(loadViewer)

type Screen = 'landing' | 'analysing' | 'viewer'

export default function App() {
  const reduced = useReducedMotion()
  const phone = usePhone()

  const [screen, setScreen] = useState<Screen>('landing')
  const [health, setHealth] = useState<Health | null>(null)
  const [samples, setSamples] = useState<Sample[] | null>(null)
  const [samplesError, setSamplesError] = useState(false)
  const [opts, setOpts] = useState<Options>({ consensus: true, clusters: true, vote: true })
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busyRetry, setBusyRetry] = useState<Source | null>(null)
  const [rerunning, setRerunning] = useState(false)
  const [srcName, setSrcName] = useState('')
  const [layers, setLayers] = useState<Layers>({ detections: true, masks: true, status: true, clusters: true, flaggedOnly: false })
  const [selected, setSelected] = useState<number | null>(null)
  const source = useRef<Source | null>(null)
  const abort = useRef<AbortController | null>(null)
  const viewer = useRef<ViewerHandle>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    getSamples().then(setSamples).catch(() => setSamplesError(true))
  }, [])

  const avail: Options = { consensus: health?.consensus_available ?? true, clusters: health?.clusters_available ?? true, vote: true }
  const effOf = (o: Options): Options => ({ consensus: o.consensus && avail.consensus, clusters: o.clusters && avail.clusters, vote: o.vote })
  const eff = effOf(opts)

  const run = useCallback(async (src: Source, o: Options, inPlace = false) => {
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
      const r = await infer(src, o, ac.signal)
      setResult(r)
      if (!inPlace) setSelected(null)
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
  const setOpt = (k: keyof Options, v: boolean) => {
    const o = { ...opts, [k]: v }
    setOpts(o)
    if (screen === 'viewer' && source.current) run(source.current, effOf(o), true)
  }
  const select = (id: number | null, zoom = false) => {
    setSelected(id)
    if (zoom && id != null) viewer.current?.zoomToCell(id)
  }

  const typical = health?.typical_ms ? health.typical_ms[eff.consensus ? 'consensus' : 'no_consensus'] : null

  return (
    <div className={`app app--${screen}`}>
      <a className="skip" href="#main">Skip to content</a>
      <header className="header">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); goNew() }}>
          <svg viewBox="0 0 32 32" className="brand__mark" aria-hidden="true">
            <ellipse cx="16" cy="16" rx="11.5" ry="10" fill="none" stroke="currentColor" strokeWidth="2.2" />
            <circle cx="17.5" cy="15" r="3.8" fill="currentColor" />
          </svg>
          <span>Cervical Cell Pipeline</span>
        </a>
        {screen !== 'landing' && <span className="header__note"><span className="header__long">Research demo · not for diagnosis</span><span className="header__short">Not for diagnosis</span></span>}
      </header>

      {screen === 'landing' && (
        <Landing samples={samples} samplesError={samplesError} opts={opts} avail={avail} setOpt={setOpt}
          error={error}
          busyRetry={busyRetry ? () => run(busyRetry, eff) : null}
          onFile={(f) => run({ kind: 'file', file: f }, eff)}
          onSample={(n) => run({ kind: 'sample', name: n }, eff)} />
      )}

      {screen === 'analysing' && (
        <Analysis name={srcName} opts={eff} typical={typical} onCancel={cancel} />
      )}

      {screen === 'viewer' && result && (
        <main id="main" className="workspace">
          <section className="stage card" aria-label="Image">
            <Toolbar layers={layers} setLayers={setLayers} opts={opts} avail={avail} setOpt={setOpt}
              clustersShown={result.clusters.used} busy={rerunning}
              onZoom={(f) => viewer.current?.zoomBy(f)} onHome={() => viewer.current?.home()} />
            {error && <p className="inline-error" role="alert">{error}</p>}
            <div className={`viewer-frame ${rerunning ? 'is-busy' : ''}`}>
              <Suspense fallback={<div className="viewer-msg" role="status">Loading viewer…</div>}>
                <Viewer ref={viewer} result={result} layers={layers} selected={selected} onSelect={(id) => select(id)}
                  reducedMotion={reduced} phone={phone} />
              </Suspense>
              {rerunning && <div className="viewer-msg" role="status">Updating…</div>}
            </div>
            <Legend clusters={layers.clusters && result.clusters.used} />
          </section>
          <SidePanel result={result} selected={selected} onSelect={select} onNew={goNew} />
        </main>
      )}
    </div>
  )
}

function Legend({ clusters }: { clusters: boolean }) {
  return (
    <ul className="legend" aria-label="Legend">
      <li className="lg lg--measurable">Measurable</li>
      <li className="lg lg--review">Review</li>
      <li className="lg lg--excluded">Excluded</li>
      <li className="lg lg--dashed">Not confirmed by YOLO26l</li>
      <li className="lg lg--not_segmented">Not segmented</li>
      {clusters && <li className="lg lg--cluster">Cluster</li>}
      {clusters && <li className="lg lg--debris">Debris</li>}
    </ul>
  )
}
