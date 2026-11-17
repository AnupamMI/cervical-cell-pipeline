import { useRef, useState } from 'react'
import { ACCEPT, checkFile, type Options, type Sample } from '../lib/api'
import { Switch } from './Switch'

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
  const [drag, setDrag] = useState(false)
  const [local, setLocal] = useState<string | null>(null)

  const take = (f: File | undefined) => {
    if (!f) return
    const e = checkFile(f)
    setLocal(e)
    if (!e) onFile(f)
  }
  const msg = local ?? error

  return (
    <main id="main" className="landing">
      <div className="landing__intro">
        <h1 className="landing__title">Cervical Cell Pipeline</h1>
        <p className="landing__lede">
          Upload one Pap-smear field of view to detect every cell, outline it and its nucleus, and see which
          masks are measurable.
        </p>
        <p className="landing__note">Research demo, not for diagnosis.</p>
      </div>

      <section className="card upload" aria-labelledby="upload-h">
        <div className={`drop ${drag ? 'is-drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files[0]) }}>
          <svg className="drop__icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 15V4m0 0L8 8m4-4 4 4M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
          </svg>
          <h2 id="upload-h" className="drop__h">Drop an image here</h2>
          <p className="drop__meta">TIFF, PNG or JPEG · 2028 × 1520 · up to 25 MB</p>
          <button type="button" className="btn btn--primary" onClick={() => input.current?.click()}>Choose image</button>
          <input ref={input} type="file" accept={ACCEPT} hidden aria-hidden="true" tabIndex={-1}
            onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
        </div>
        <div className="upload__opts">
          <Switch checked={opts.consensus && avail.consensus} disabled={!avail.consensus}
            onChange={(v) => setOpt('consensus', v)} label="Consensus check"
            hint={avail.consensus ? 'YOLO26l confirms each cell. Never removes cells.' : 'Not available on this deployment.'} />
          <Switch checked={opts.clusters && avail.clusters} disabled={!avail.clusters}
            onChange={(v) => setOpt('clusters', v)} label="Clusters & debris (detector C)"
            hint={avail.clusters ? 'Outlines clusters and debris (never segmented). Its cell class is the third vote.' : 'Not available on this deployment.'} />
          <Switch checked={opts.vote} onChange={(v) => setOpt('vote', v)} label="Segment voted cells only"
            hint="Only cells at least 2 of 3 detectors agree on, outside clusters and debris. Others are shown, not segmented." />
        </div>
        {msg && <p className="error" role="alert">{msg}</p>}
        {busyRetry && (
          <p className="error error--busy" role="status">
            <span>Another image is being analysed.</span>
            <button type="button" className="btn btn--quiet btn--sm" onClick={busyRetry}>Retry</button>
          </p>
        )}
      </section>

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

      <p className="landing__foot muted small">Images are deleted after decoding; results after 15 minutes.</p>
    </main>
  )
}
