import { useEffect, useState } from 'react'
import { useThree } from '@react-three/fiber'
import type * as THREE from 'three'

/**
 * Compile every shader in the scene before the first frame, without blocking the main thread on the link step:
 * gl.compile() starts compilation (with KHR_parallel_shader_compile the driver compiles in the background), then
 * we poll each program's isReady() once per frame. Until then the canvas stays on frameloop "never".
 *
 * Same idea as WebGLRenderer.compileAsync, which is not used because it throws inside its own rAF loop
 * ("reading 'isReady'") when a material is replaced or disposed between compile and poll (ShaderGradient re-creates
 * its material after mount). Here a replaced material simply counts as done. Capped at 4 s.
 */
export function useCompiled(onReady: () => void) {
  const { gl, scene, camera } = useThree()
  useEffect(() => {
    let raf = 0
    let cancelled = false
    const t0 = performance.now()
    raf = requestAnimationFrame(() => {
      let mats: Set<THREE.Material>
      try {
        mats = gl.compile(scene, camera) as unknown as Set<THREE.Material>
      } catch {
        onReady()
        return
      }
      const props = (gl as unknown as { properties: { get(m: THREE.Material): { currentProgram?: { isReady(): boolean } } } }).properties
      const poll = () => {
        if (cancelled) return
        let pending = 0
        for (const m of mats) {
          const prog = props.get(m)?.currentProgram
          if (prog && !prog.isReady()) pending++
        }
        if (pending === 0 || performance.now() - t0 > 4000) onReady()
        else raf = requestAnimationFrame(poll)
      }
      raf = requestAnimationFrame(poll)
    })
    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
    }
  }, [gl, scene, camera, onReady])
}

export function Compile({ onReady }: { onReady: () => void }) {
  useCompiled(onReady)
  return null
}

export function useFrameloop(active: boolean) {
  const [compiled, setCompiled] = useState(false)
  return { frameloop: (active && compiled ? 'always' : 'never') as 'always' | 'never', compiled, setCompiled }
}
