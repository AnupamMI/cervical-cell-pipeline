import { useRef, useState } from 'react'
import { ACCEPT, checkFile, type Sample } from '../lib/api'
import { Glass } from './Glass'
import { Switch } from './Switch'

export function Landing({
  samples, samplesError, onFile, onSample, consensus, setConsensus, consensusAvailable, error, cellSlot,
}: {
  samples: Sample[] | null
  samplesError: boolean
  onFile: (f: File) => void
  onSample: (name: string) => void
  consensus: boolean
  setConsensus: (v: boolean) => void
  consensusAvailable: boolean
  error: string | null
  cellSlot: React.ReactNode
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
      <section className="landing__hero">
        <p className="eyebrow">Pap-smear cytology · single field of view</p>
        <h1 className="landing__title">Cervical Cell Pipeline</h1>
        <p className="landing__subtitle">Research demo, not for diagnosis</p>
        <p className="landing__lede">
          Upload one 2028&thinsp;×&thinsp;1520 FOV. YOLO26s finds the cells, YOLO26l cross-checks them, a TinyUNet
          outlines each cell and its nucleus, and a quality gate marks every mask as measurable, review or excluded.
        </p>

        <Glass as="section" variant="text" radius={26} className={`upload ${drag ? 'is-drag' : ''}`}
          aria-labelledby="upload-h"
          onDragOver={(e: React.DragEvent) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e: React.DragEvent) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files[0]) }}>
          <div className="upload__drop">
            <svg className="upload__icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
            </svg>
            <h2 id="upload-h" className="upload__h">Drop an FOV image here</h2>
            <p className="upload__meta">TIFF, PNG or JPEG · up to 25 MB · deleted after 15 minutes</p>
            <button type="button" className="btn btn--primary" onClick={() => input.current?.click()}>
              Choose an image
            </button>
            <input ref={input} type="file" accept={ACCEPT} hidden
              onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
          </div>
          <div className="upload__opts">
            <Switch checked={consensus && consensusAvailable} disabled={!consensusAvailable} onChange={setConsensus}
              label="Consensus check (YOLO26l)"
              hint={consensusAvailable ? 'Tags each cell as confirmed or YOLO26s-only. Never removes cells.' : 'Not available on this deployment.'} />
          </div>
          {msg && <p className="upload__error" role="alert">{msg}</p>}
        </Glass>

        <section className="samples" aria-labelledby="samples-h">
          <h2 id="samples-h" className="samples__h">Try a sample</h2>
          {samplesError && <p className="muted">Samples could not be loaded.</p>}
          <div className="samples__row">
            {samples === null && !samplesError &&
              Array.from({ length: 4 }, (_, i) => <span key={i} className="sample sample--skeleton" aria-hidden="true" />)}
            {samples?.slice(0, 4).map((s, i) => (
              <button key={s.name} type="button" className="sample" onClick={() => onSample(s.name)}
                aria-label={`Analyse sample ${i + 1} (${s.name})`}>
                <img src={s.thumb} alt="" width={120} height={90} loading="lazy" decoding="async" />
                <span>Sample {i + 1}</span>
              </button>
            ))}
          </div>
        </section>
      </section>
      <div className="landing__visual">{cellSlot}</div>
    </main>
  )
}
