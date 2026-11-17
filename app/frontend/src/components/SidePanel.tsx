import { useEffect, useMemo, useRef, useState } from 'react'
import type { Cell, Result, Status } from '../lib/api'
import { C_NOTE, DET_LABEL, STATUS_HINT, STATUS_LABEL, fmtMs, reason, voterNames } from '../lib/labels'

type Key = 'id' | 'status' | 'conf' | 'votes' | 'cell_area_px' | 'nc_ratio'
const COLS: { key: Key; label: string; num?: boolean; title?: string }[] = [
  { key: 'id', label: '#', num: true },
  { key: 'status', label: 'Status' },
  { key: 'conf', label: 'Conf', num: true },
  { key: 'votes', label: 'Votes', title: 'Detectors that found this cell (YOLO26s, YOLO26l, detector C)' },
  { key: 'nc_ratio', label: 'N:C', num: true },
]
const STATUS_ORDER: Record<Status, number> = { not_segmented: 0, excluded: 1, review: 2, measurable: 3 }

export function StatusChip({ s }: { s: Status }) {
  return (
    <span className={`chip chip--${s}`}>
      <span className="chip__mark" aria-hidden="true" />
      {STATUS_LABEL[s]}
    </span>
  )
}

function sortVal(c: Cell, k: Key) {
  if (k === 'status') return STATUS_ORDER[c.status]
  if (k === 'nc_ratio') return c.nc_ratio ?? -1
  return c[k] as number | string
}

export function SidePanel({ result, selected, onSelect, onNew }: {
  result: Result
  selected: number | null
  onSelect: (id: number, zoom: boolean) => void
  onNew: () => void
}) {
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: 'id', dir: 1 })
  const rows = useMemo(() => {
    const r = [...result.cells]
    r.sort((a, b) => {
      const x = sortVal(a, sort.key), y = sortVal(b, sort.key)
      return (x < y ? -1 : x > y ? 1 : a.id - b.id) * sort.dir
    })
    return r
  }, [result.cells, sort])
  const body = useRef<HTMLTableSectionElement>(null)
  useEffect(() => {
    // keep the selected row visible by scrolling the table only (scrollIntoView would also scroll the page)
    const row = body.current?.querySelector<HTMLTableRowElement>('[aria-selected="true"]')
    const wrap = row?.closest<HTMLElement>('.tablewrap')
    if (!row || !wrap) return
    const head = wrap.querySelector('thead')?.getBoundingClientRect().height ?? 0
    const top = row.offsetTop - head, bottom = row.offsetTop + row.offsetHeight - wrap.clientHeight
    if (wrap.scrollTop > top) wrap.scrollTop = top
    else if (wrap.scrollTop < bottom) wrap.scrollTop = bottom
  }, [selected])

  const st = result.counts.status
  const reasons = Object.entries(result.counts.reasons).slice(0, 5)
  const sel = result.cells.find((c) => c.id === selected)
  const qcIssues = [...result.image_qc.flags.map(reason), ...result.warnings]
  const t = result.timings_ms
  const cl = result.clusters
  const exp = new Date(result.expires_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <aside className="panel card" aria-label="Results">
      <div className="panel__head">
        <div>
          <p className="eyebrow">{result.source.sample ? `Sample ${result.source.sample}` : 'Your image'}</p>
          <h2 className="panel__title">{result.counts.total} cells</h2>
        </div>
        <button type="button" className="btn btn--quiet btn--sm" onClick={onNew}>New image</button>
      </div>

      <ul className="stats" aria-label="Cells by status">
        {(['measurable', 'review', 'excluded', 'not_segmented'] as Status[]).map((s) => (
          <li key={s} className={`stat stat--${s}`} title={STATUS_HINT[s]}>
            <span className="stat__n">{st[s] ?? 0}</span>
            <span className="stat__label"><span className="dot" aria-hidden="true" />{STATUS_LABEL[s]}</span>
          </li>
        ))}
      </ul>

      {qcIssues.length > 0 && (
        <div className="notice" role="note">
          <strong>Image check</strong>
          <ul>{qcIssues.map((q) => <li key={q}>{q}</li>)}</ul>
        </div>
      )}

      {cl.used && (
        <section className="group" aria-labelledby="c-h">
          <h3 id="c-h" className="group__h">Clusters &amp; debris <span className="badge">Detector C</span></h3>
          <dl className="kv">
            <div><dt><span className="sw sw--cluster" aria-hidden="true" />Clusters</dt><dd>{result.counts.clusters.cluster}</dd></div>
            <div><dt><span className="sw sw--debris" aria-hidden="true" />Debris</dt><dd>{result.counts.clusters.debris}</dd></div>
            <div><dt>Cells near a cluster</dt><dd>{result.counts.near.cluster}</dd></div>
            <div><dt>Cells near debris</dt><dd>{result.counts.near.debris}</dd></div>
          </dl>
          <p className="small muted">{result.voting.used
            ? 'Clusters and debris are never segmented; cells inside them are skipped. Detector C is experimental (mAP50 ~0.5).'
            : C_NOTE}</p>
        </section>
      )}

      {reasons.length > 0 && (
        <section className="group" aria-labelledby="reasons-h">
          <h3 id="reasons-h" className="group__h">Top reasons</h3>
          <dl className="kv">
            {reasons.map(([k, n]) => <div key={k}><dt>{reason(k)}</dt><dd>{n}</dd></div>)}
          </dl>
        </section>
      )}

      {sel && (
        <section className="group selcell" aria-live="polite" aria-label={`Cell ${sel.id}`}>
          <div className="selcell__top">
            <h3 className="group__h">Cell {sel.id}</h3>
            <StatusChip s={sel.status} />
          </div>
          {sel.reasons.length > 0 && <p className="selcell__why">{sel.reasons.map(reason).join(' · ')}</p>}
          <dl className="kv kv--grid">
            <div><dt>Confidence</dt><dd>{sel.conf.toFixed(3)}</dd></div>
            <div><dt>Votes</dt><dd>{sel.votes} of {result.voting.detectors_run}</dd></div>
            <div className="kv__wide"><dt>Found by</dt><dd>{voterNames(sel.voters)}</dd></div>
            <div className="kv__wide"><dt>Detection</dt><dd>{DET_LABEL[sel.det_tag]}</dd></div>
            {sel.status !== 'not_segmented' && <>
              <div><dt>Cell area</dt><dd>{sel.cell_area_px.toLocaleString()} px</dd></div>
              <div><dt>Nucleus</dt><dd>{sel.nucleus_area_px.toLocaleString()} px</dd></div>
              <div><dt>N:C</dt><dd>{sel.nc_ratio == null ? '—' : sel.nc_ratio.toFixed(3)}</dd></div>
              <div><dt>Occluded</dt><dd>{Math.round(sel.occluded_frac * 100)}%</dd></div>
            </>}
            {cl.used && !result.voting.used && (sel.near_cluster.length > 0 || sel.near_debris.length > 0) && (
              <div className="kv__wide"><dt>Detector C (info)</dt>
                <dd>{[sel.near_cluster.length ? 'near a cluster' : '', sel.near_debris.length ? 'near debris' : ''].filter(Boolean).join(', ')}</dd></div>
            )}
          </dl>
        </section>
      )}

      <section className="group cells" aria-labelledby="cells-h">
        <h3 id="cells-h" className="group__h">Cells <span className="muted">· select a row to zoom</span></h3>
        <div className="tablewrap" tabIndex={-1}>
          <table className="table">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th key={c.key} scope="col" className={`${c.num ? 'num' : ''} col-${c.key}`} title={c.title}
                    aria-sort={sort.key === c.key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
                    <button type="button" onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? (-s.dir as 1 | -1) : 1 }))}>
                      {c.label}
                      <span className="sortmark" aria-hidden="true">{sort.key === c.key ? (sort.dir === 1 ? '↑' : '↓') : ''}</span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody ref={body}>
              {rows.length === 0 && (
                <tr><td colSpan={COLS.length} className="muted empty">No cells were detected in this image.</td></tr>
              )}
              {rows.map((c) => (
                <tr key={c.id} aria-selected={c.id === selected} className={c.id === selected ? 'is-sel' : ''}
                  onClick={() => onSelect(c.id, true)}>
                  <td className="num">
                    <button type="button" className="rowbtn" onClick={(e) => { e.stopPropagation(); onSelect(c.id, true) }}
                      aria-label={`Zoom to cell ${c.id}, ${STATUS_LABEL[c.status]}`}>{c.id}</button>
                  </td>
                  <td><StatusChip s={c.status} /></td>
                  <td className="num">{c.conf.toFixed(2)}</td>
                  <td><span className={`votes votes--${c.votes}`} title={voterNames(c.voters)}>
                    {c.votes}/{result.voting.detectors_run}<span className="sr-only"> ({voterNames(c.voters)})</span></span></td>
                  <td className="num">{c.nc_ratio == null ? '—' : c.nc_ratio.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="group" aria-labelledby="dl-h">
        <h3 id="dl-h" className="group__h">Downloads</h3>
        <div className="downloads">
          <a className="btn btn--quiet btn--sm" href={result.downloads['overlay.png']} download>Overlay PNG</a>
          <a className="btn btn--quiet btn--sm" href={result.downloads['cells.csv']} download>cells.csv</a>
          <a className="btn btn--quiet btn--sm" href={result.downloads['coco.json']} download>COCO JSON</a>
        </div>
        <p className="small muted">Kept until {exp}, then deleted.</p>
      </section>

      <p className="timings small muted">
        Detect {fmtMs(t.detect)}
        {result.consensus.used ? ` · consensus ${fmtMs(t.consensus)}` : ''}
        {cl.used ? ` · clusters ${fmtMs(t.clusters)}` : ''} · segment {fmtMs(t.segment)} · gate {fmtMs(t.gate)} · total {fmtMs(t.request)}
      </p>
    </aside>
  )
}
