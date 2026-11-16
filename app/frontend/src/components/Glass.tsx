import type { CSSProperties, ElementType, ReactNode } from 'react'

/**
 * Frosted "liquid glass" surface.
 *
 * Visual model ported from Liquid Glass JS (github.com/dashersw/liquid-glass-js, MIT, (c) 2025 Armagan Amcalar):
 * its shader composes a blurred backdrop (13x13 Gaussian, u_blurRadius), a tint (u_tintOpacity, default 0.2) and
 * three light layers measured from the shape edge: a base sheen (u_baseIntensity), a bright rim (u_rimIntensity /
 * u_rimDistance) and an edge refraction band (u_edgeIntensity), plus a corner boost (u_cornerBoost).
 * Here the same layers are CSS: backdrop-filter = blur + tint, ::before = rim + edge band, ::after = base sheen +
 * corner boost. The library itself samples a one-off html2canvas snapshot of the page, which cannot see the WebGL
 * background (and would freeze it) and costs a full-page capture on load; CSS backdrop-filter refracts the live
 * gradient for free. See README > Frontend for the trade-off.
 *
 * variant "text": a darker fill behind the glass so body text keeps >= 4.5:1 contrast on any gradient frame.
 */
export function Glass({
  as: Tag = 'div',
  variant = 'chrome',
  radius = 22,
  tint = 0.2,
  className = '',
  style,
  children,
  ...rest
}: {
  as?: ElementType
  variant?: 'chrome' | 'text'
  radius?: number
  tint?: number
  className?: string
  style?: CSSProperties
  children?: ReactNode
} & Record<string, unknown>) {
  return (
    <Tag
      className={`glass glass--${variant} ${className}`}
      style={{ '--glass-r': `${radius}px`, '--glass-tint': tint, ...style } as CSSProperties}
      {...rest}
    >
      {children}
    </Tag>
  )
}
