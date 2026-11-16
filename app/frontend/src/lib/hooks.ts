import { useEffect, useState } from 'react'

function useMedia(query: string) {
  const [v, setV] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches)
  useEffect(() => {
    const m = window.matchMedia(query)
    const on = () => setV(m.matches)
    on()
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [query])
  return v
}

export const useReducedMotion = () => useMedia('(prefers-reduced-motion: reduce)')
/** Phones / small touch screens get lighter effects (lower pixel density, simpler geometry, no pointer tilt). */
export const usePhone = () => useMedia('(max-width: 760px), (pointer: coarse)')

export function usePageVisible() {
  const [v, setV] = useState(() => typeof document === 'undefined' || !document.hidden)
  useEffect(() => {
    const on = () => setV(!document.hidden)
    document.addEventListener('visibilitychange', on)
    return () => document.removeEventListener('visibilitychange', on)
  }, [])
  return v
}

/**
 * true after the first user interaction (pointer move / down, touch, wheel, scroll, key). The landing's WebGL
 * effects start here: first paint is the static still frame, so the page is complete and interactive at once,
 * and the one-off WebGL start-up (context, programs, ShaderGradient set-up) never competes with loading.
 */
export function useFirstInteraction() {
  const [on, setOn] = useState(false)
  useEffect(() => {
    if (on) return
    const evs = ['pointermove', 'pointerdown', 'touchstart', 'wheel', 'scroll', 'keydown'] as const
    const go = () => setOn(true)
    evs.forEach((e) => window.addEventListener(e, go, { once: true, passive: true }))
    return () => evs.forEach((e) => window.removeEventListener(e, go))
  }, [on])
  return on
}

/** value becomes true `ms` after `trigger` became true (stays true). */
export function useAfter(trigger: boolean, ms: number) {
  const [on, setOn] = useState(false)
  useEffect(() => {
    if (!trigger || on) return
    const t = window.setTimeout(() => setOn(true), ms)
    return () => clearTimeout(t)
  }, [trigger, on, ms])
  return on
}
