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
export const usePhone = () => useMedia('(max-width: 760px), (pointer: coarse)')
