import type { Status } from './api'

export const STATUS_LABEL: Record<Status, string> = {
  measurable: 'Measurable',
  review: 'Review',
  excluded: 'Excluded',
}

export const STATUS_HINT: Record<Status, string> = {
  measurable: 'Mask passed every check',
  review: 'Usable, but worth a look',
  excluded: 'Mask not usable for measurement',
}

// Gate reason codes (ncmask/quality.py) and image QC flags in plain words
export const REASON: Record<string, string> = {
  cell_at_edge: 'Cut by the image edge',
  box_too_small: 'Detection too small to segment',
  empty_mask: 'No cell mask found',
  no_nucleus: 'No nucleus found',
  implausible_mask: 'Implausible size or N:C',
  multi_nucleus: 'Several nuclei',
  occluded: 'Mostly covered by other cells',
  s_only_low_conf: 'Low confidence, not confirmed by YOLO26l',
  nucleus_touches_cell_boundary: 'Nucleus touches the cell outline',
  likely_inflammatory: 'Small: likely inflammatory cell',
  likely_cluster: 'Large: likely a cluster',
  blurry: 'Image looks blurry',
  too_dark: 'Image is too dark',
  too_bright: 'Image is too bright',
  dark_regions: 'Large dark regions',
  low_stain: 'Very little stained material',
  heavy_stain: 'Very heavy staining',
}

export const reason = (code: string) => REASON[code] ?? code.replace(/_/g, ' ')

export const DET_LABEL: Record<string, string> = {
  consensus: 'Confirmed by YOLO26l',
  s_only: 'YOLO26s only',
  single: 'Consensus check off',
}

export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)
