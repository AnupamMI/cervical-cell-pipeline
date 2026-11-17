export type Status = 'measurable' | 'review' | 'excluded' | 'not_segmented'
export type DetTag = 'consensus' | 's_only' | 'single' | 'bc_only'
export type Pt = [number, number]

export interface Cell {
  id: number
  box: [number, number, number, number]
  conf: number
  det_tag: DetTag
  b_conf: number | null
  ab_iou: number | null
  status: Status
  reasons: string[]
  flags: string[]
  cell_area_px: number
  nucleus_area_px: number
  nc_ratio: number | null
  occluded_frac: number
  votes: number
  voters: string
  near_cluster: string[]
  near_debris: string[]
  cell_polygons: Pt[][]
  nucleus_polygons: Pt[][]
}

export type CKind = 'cluster' | 'debris'
export interface CBox {
  id: string
  class: CKind
  conf: number
  box: [number, number, number, number]
}

export interface Result {
  id: string
  expires_utc: string
  size: { width: number; height: number; expected: [number, number]; is_expected: boolean }
  image_qc: { ok: boolean; flags: string[]; metrics: Record<string, number> }
  consensus: { requested: boolean; used: boolean; l_only: number }
  clusters: { requested: boolean; used: boolean; info_only: true; conf: number | null; near_frac: number | null; boxes: CBox[] }
  voting: { used: boolean; detectors_run: number; votes_needed: number | null; skip_in_cluster_debris: boolean; bc_only: number }
  cells: Cell[]
  counts: {
    total: number
    status: Record<Status, number>
    reasons: Record<string, number>
    det_tag: Record<string, number>
    dropped: Record<string, number>
    votes: Record<string, number>
    clusters: Record<CKind, number>
    near: Record<CKind, number>
  }
  timings_ms: Record<'detect' | 'consensus' | 'clusters' | 'segment' | 'gate' | 'total' | 'decode' | 'request', number>
  warnings: string[]
  downloads: Record<'overlay.png' | 'cells.csv' | 'coco.json' | 'image.jpg', string>
  source: { sample?: string; upload?: boolean }
}

export interface Sample {
  name: string
  file: string
  bytes: number
  thumb: string
}

export interface Health {
  status: 'ok' | 'warming_up'
  busy: boolean
  consensus_available: boolean
  clusters_available: boolean
  max_upload_bytes: number
  typical_ms: null | Record<'consensus' | 'no_consensus', Record<string, number>>
  models: Record<string, { source: string; onnx_sha256: string; role: string }>
}

export class ApiError extends Error {
  constructor(public code: string, message: string, public status: number) {
    super(message)
  }
}

async function json<T>(r: Response): Promise<T> {
  let body: unknown = null
  try {
    body = await r.json()
  } catch {
    /* non-JSON error page */
  }
  if (!r.ok) {
    const b = (body ?? {}) as { error?: string; detail?: unknown }
    const msg = typeof b.detail === 'string' ? b.detail : r.status === 413 ? 'File is too large.' : `Request failed (${r.status}).`
    throw new ApiError(b.error ?? 'http_error', msg, r.status)
  }
  return body as T
}

export const getHealth = () => fetch('/api/health').then((r) => json<Health>(r))
export const getSamples = () => fetch('/api/samples').then((r) => json<{ samples: Sample[] }>(r)).then((x) => x.samples)

export type Source = { kind: 'file'; file: File } | { kind: 'sample'; name: string }

export interface Options {
  consensus: boolean
  clusters: boolean
  vote: boolean
}

export async function infer(src: Source, o: Options, signal?: AbortSignal): Promise<Result> {
  const fd = new FormData()
  if (src.kind === 'file') fd.append('file', src.file)
  else fd.append('sample', src.name)
  const r = await fetch(`/api/infer?consensus=${o.consensus}&clusters=${o.clusters}&vote=${o.vote}`, { method: 'POST', body: fd, signal })
  return json<Result>(r)
}

export const MAX_BYTES = 25 * 1024 * 1024
export const ACCEPT = '.tif,.tiff,.png,.jpg,.jpeg'
export function checkFile(f: File): string | null {
  if (!/\.(tiff?|png|jpe?g)$/i.test(f.name)) return 'Unsupported file type. Use .tif, .tiff, .png, .jpg or .jpeg.'
  if (f.size > MAX_BYTES) return `File is ${(f.size / 1048576).toFixed(1)} MB; the limit is 25 MB.`
  return null
}
