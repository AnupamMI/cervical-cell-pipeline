import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float, PerformanceMonitor } from '@react-three/drei'
import * as THREE from 'three'
import { mergeVertices } from 'three-stdlib'
import { Compile, useFrameloop } from './useCompiled'

/**
 * One gentle 3D moment for the landing screen (React Three Fiber + drei, MIT): a slowly rotating translucent
 * squamous cell (flattened, slightly irregular cytoplasm, darker nucleus, a few granules) lit pink/indigo, tilting a
 * little toward the mouse. active=false -> frameloop "never" (no frames at all): the parent passes false when the tab
 * is hidden or an image is loaded; reduced motion never mounts this (static SVG instead).
 */

function noise3(x: number, y: number, z: number) {
  // smooth, deterministic pseudo-noise (sum of sines): enough for an organic outline, no dependency
  return (
    Math.sin(x * 1.7 + y * 0.9) * 0.5 +
    Math.sin(y * 2.3 - z * 1.1 + 1.3) * 0.3 +
    Math.sin(z * 2.9 + x * 1.4 + 2.1) * 0.2
  )
}

function useCellGeometry(detail: number) {
  return useMemo(() => {
    // Icosahedron geometry is non-indexed (one vertex copy per face): merge first, or the displacement tears the
    // surface and computeVertexNormals() gives flat, faceted shading
    const base = new THREE.IcosahedronGeometry(1.55, detail)
    base.deleteAttribute('normal')
    base.deleteAttribute('uv')
    const g = mergeVertices(base)
    base.dispose()
    const p = g.attributes.position as THREE.BufferAttribute
    const v = new THREE.Vector3()
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i)
      const n = v.clone().normalize()
      const r = 1 + 0.09 * noise3(n.x * 2, n.y * 2, n.z * 2)
      v.multiplyScalar(r)
      v.y *= 0.42 // squamous cells are flat
      p.setXYZ(i, v.x, v.y, v.z)
    }
    g.computeVertexNormals()
    return g
  }, [detail])
}

const GRANULES = Array.from({ length: 26 }, (_, i) => {
  const a = i * 2.39996, r = 0.55 + 0.75 * ((i * 0.618) % 1)
  return [Math.cos(a) * r, (((i * 0.37) % 1) - 0.5) * 0.25, Math.sin(a) * r * 0.9] as const
})

function Cell({ lite, tilt }: { lite: boolean; tilt: React.RefObject<{ x: number; y: number }> }) {
  const group = useRef<THREE.Group>(null)
  const geo = useCellGeometry(lite ? 4 : 6)
  const granules = useRef<THREE.InstancedMesh>(null)

  useEffect(() => {
    const m = granules.current
    if (!m) return
    const o = new THREE.Object3D()
    GRANULES.forEach(([x, y, z], i) => {
      o.position.set(x, y, z)
      o.scale.setScalar(0.028 + 0.02 * ((i * 0.53) % 1))
      o.updateMatrix()
      m.setMatrixAt(i, o.matrix)
    })
    m.instanceMatrix.needsUpdate = true
  }, [])

  useFrame((_, dt) => {
    const g = group.current
    if (!g) return
    const d = Math.min(dt, 0.05)
    g.rotation.y += d * 0.12
    const t = tilt.current ?? { x: 0, y: 0 }
    g.rotation.x += (0.38 + t.y * 0.22 - g.rotation.x) * Math.min(1, d * 2.5)
    g.rotation.z += (-t.x * 0.18 - g.rotation.z) * Math.min(1, d * 2.5)
  })

  return (
    <Float speed={0.8} rotationIntensity={0.05} floatIntensity={0.25}>
      <group ref={group} rotation={[0.38, 0, 0]} scale={0.78}>
        {/* soft halo */}
        <mesh geometry={geo} scale={1.06}>
          <meshBasicMaterial color="#F6C1D9" transparent opacity={0.07} side={THREE.BackSide} depthWrite={false}
            blending={THREE.AdditiveBlending} />
        </mesh>
        {/* cytoplasm */}
        <mesh geometry={geo}>
          <meshStandardMaterial color="#f6c1d9" emissive="#c2588f" emissiveIntensity={0.18} roughness={0.32}
            metalness={0.05} transparent opacity={0.26} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
        {/* nucleus */}
        <mesh position={[0.28, 0.02, 0.12]} scale={[1, 0.8, 1]}>
          <sphereGeometry args={[0.4, lite ? 24 : 48, lite ? 16 : 32]} />
          <meshStandardMaterial color="#2b2275" emissive="#4B3FB3" emissiveIntensity={0.25} roughness={0.55} />
        </mesh>
        <instancedMesh ref={granules} args={[undefined, undefined, GRANULES.length]}>
          <sphereGeometry args={[1, 8, 6]} />
          <meshStandardMaterial color="#fde7f1" emissive="#F6C1D9" emissiveIntensity={0.35} transparent opacity={0.7} />
        </instancedMesh>
      </group>
    </Float>
  )
}

export default function CellScene({ active, lite }: { active: boolean; lite: boolean }) {
  const tilt = useRef({ x: 0, y: 0 })
  const [dpr, setDpr] = useState(lite ? 1 : 1.5)
  const { frameloop, compiled, setCompiled } = useFrameloop(active)
  const ready = useCallback(() => setCompiled(true), [setCompiled])

  useEffect(() => {
    if (lite) return
    const on = (e: PointerEvent) => {
      tilt.current = { x: (e.clientX / window.innerWidth) * 2 - 1, y: (e.clientY / window.innerHeight) * 2 - 1 }
    }
    window.addEventListener('pointermove', on, { passive: true })
    return () => window.removeEventListener('pointermove', on)
  }, [lite])

  return (
    <Canvas
      className={`fx-cell__canvas ${compiled ? 'is-ready' : ''}`}
      frameloop={frameloop}
      dpr={dpr}
      camera={{ position: [0, 0.4, 4.6], fov: 38 }}
      gl={{ antialias: !lite, alpha: true, powerPreference: 'low-power' }}
      aria-hidden="true"
    >
      <Compile onReady={ready} />
      <PerformanceMonitor onDecline={() => setDpr(1)} />
      <ambientLight intensity={0.45} />
      <pointLight position={[3, 2.5, 3.5]} intensity={28} color="#F6C1D9" />
      <pointLight position={[-3.2, -1.2, 2.2]} intensity={22} color="#6f62e0" />
      <directionalLight position={[0, 4, 2]} intensity={0.6} />
      <Cell lite={lite} tilt={tilt} />
    </Canvas>
  )
}
