/** Static cell illustration: shown before the 3D scene loads, and instead of it under reduced motion / no WebGL. */
export function StaticCell() {
  return (
    <svg className="fx-cell__static" viewBox="0 0 400 400" aria-hidden="true">
      <defs>
        <radialGradient id="cyto" cx="0.42" cy="0.38" r="0.7">
          <stop offset="0" stopColor="#fbe1ec" stopOpacity="0.55" />
          <stop offset="0.65" stopColor="#F6C1D9" stopOpacity="0.28" />
          <stop offset="1" stopColor="#4B3FB3" stopOpacity="0.18" />
        </radialGradient>
        <radialGradient id="nuc" cx="0.4" cy="0.35" r="0.75">
          <stop offset="0" stopColor="#6c60d6" />
          <stop offset="1" stopColor="#241c66" />
        </radialGradient>
      </defs>
      <path d="M200 92c58 2 118 30 124 92 6 58-46 112-122 118-80 6-132-30-136-96-4-62 60-116 134-114z"
        fill="url(#cyto)" stroke="#F6C1D9" strokeOpacity="0.35" />
      <ellipse cx="218" cy="190" rx="38" ry="31" fill="url(#nuc)" />
    </svg>
  )
}
