import { useEffect, useMemo, useRef, useState } from 'react'
import type { Cell, Result, Status } from '../lib/api'
import { DET_LABEL, STATUS_HINT, STATUS_LABEL, fmtMs, reason } from '../lib/labels'
import { Glass } from './Glass'

type Key = 'id' | 'status' | 'conf' | 'det_tag' | 'cell_area_px' | 'nc_ratio'
const COLS: { key: Key; label: string; num?: boolean; title?: string }[] = [
  { key: 'id', label: '#', num: true },
  { key: 'status', label: 'Status' },
  { key: 'conf', label: 'Conf', num: true },
  { key: 'det_tag', label: '26l', title: 'Confirmed by the YOLO26l consensus check' },
  { key: 'cell_area_px', label: 'Area', num: true, title: 'Cell area in pixels' },
  { key: 'nc_ratio', label: 'N:C', num: true },
]
const STATUS_ORDER: Record<Status, number> = { excluded: 0, review: 1, measurable: 2 }

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
    // keep the selected row visible by scrolling the table only (scrollIntoView would also scroll the panel/page)
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

  return (
    <Glass as="aside" variant="text" radius={24} className="panel" aria-label="Results">
      <div className="panel__head">
        <div>
          <p className="eyebrow">{result.source.sample ? `Sample ${result.source.sample}` : 'Your upload'}</p>
          <h2 className="panel__title">{result.counts.total} cells</h2>
        </div>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onNew}>New image</button>
      </div>

      <ul className="counts" aria-label="Cells by status">
        {(['measurable', 'review', 'excluded'] as Status[]).map((s) => (
          <li key={s} className={`count count--${s}`} title={STATUS_HINT[s]}>
            <span className="count__n">{st[s] ?? 0}</span>
            <StatusChip s={s} />
          </li>
        ))}
      </ul>

      <section className="downloads" aria-labelledby="dl-h">
        <h3 id="dl-h" className="panel__h">Downloads</h3>
        <div className="downloads__row">
          <a className="btn btn--sm" href={result.downloads['overlay.png']} download>Overlay PNG</a>
          <a className="btn btn--sm" href={result.downloads['cells.csv']} download>cells.csv</a>
          <a className="btn btn--sm" href={result.downloads['coco.json']} download>COCO</a>
        </div>
        <p className="muted small">Kept on the server until {new Date(result.expires_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}, then deleted.</p>
      </section>

      {qcIssues.length > 0 && (
        <div className="qc" role="note">
          <strong>Image check</strong>
          <ul>{qcIssues.map((q) => <li key={q}>{q}</li>)}</ul>
        </div>
      )}

      {reasons.length > 0 && (
        <section className="reasons" aria-labelledby="reasons-h">
          <h3 id="reasons-h" className="panel__h">Top reasons</h3>
          <ul>
            {reasons.map(([k, n]) => (
              <li key={k}><span>{reason(k)}</span><span className="reasons__n">{n}</span></li>
            ))}
          </ul>
        </section>
      )}

      {sel && (
        <section className="selcell" aria-live="polite" aria-label={`Cell ${sel.id}`}>
          <div className="selcell__top">
            <h3 className="panel__h">Cell {sel.id}</h3>
            <StatusChip s={sel.status} />
          </div>
          {sel.reasons.length > 0 && <p className="selcell__why">{sel.reasons.map(reason).join(' · ')}</p>}
          <dl className="selcell__dl">
            <div><dt>Confidence</dt><dd>{sel.conf.toFixed(3)}</dd></div>
            <div><dt>Detection</dt><dd>{DET_LABEL[sel.det_tag]}</dd></div>
            <div><dt>Cell area</dt><dd>{sel.cell_area_px.toLocaleString()} px</dd></div>
            <div><dt>Nucleus</dt><dd>{sel.nucleus_area_px.toLocaleString()} px</dd></div>
            <div><dt>N:C</dt><dd>{sel.nc_ratio == null ? '—' : sel.nc_ratio.toFixed(3)}</dd></div>
            <div><dt>Occluded</dt><dd>{Math.round(sel.occluded_frac * 100)}%</dd></div>
          </dl>
        </section>
      )}

      <section className="cells" aria-labelledby="cells-h">
        <h3 id="cells-h" className="panel__h">Cells <span className="muted">· select a row to zoom</span></h3>
        <div className="tablewrap" tabIndex={-1}>
          <table className="table">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th key={c.key} scope="col" className={`${c.num ? 'num' : ''} col-${c.key}`} title={c.title}
                    aria-sort={sort.key === c.key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
                    <button type="button" onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? (-s.dir as 1 | -1) : 1 }))}>
                      {c.label}
                      <span className="sortmark" aria-hidden="true">{sort.key === c.key ? (sort.dir === 1 ? '▲' : '▼') : ''}</span>
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
                  <td><span className={`tag tag--${c.det_tag}`} title={DET_LABEL[c.det_tag]}>
                    <span aria-hidden="true">{c.det_tag === 'consensus' ? '✓' : c.det_tag === 's_only' ? '–' : 'off'}</span>
                    <span className="sr-only">{DET_LABEL[c.det_tag]}</span></span></td>
                  <td className="num col-cell_area_px">{c.cell_area_px.toLocaleString()}</td>
                  <td className="num">{c.nc_ratio == null ? '—' : c.nc_ratio.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>


      <p className="timings muted small">
        detect {fmtMs(t.detect)}
        {result.consensus.used ? ` · consensus ${fmtMs(t.consensus)}` : ''} · segment {fmtMs(t.segment)} · gate {fmtMs(t.gate)} · total {fmtMs(t.request)}
      </p>
    </Glass>
  )
}
