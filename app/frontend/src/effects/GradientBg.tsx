import { useCallback, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { ShaderGradient } from '@shadergradient/react'
import { ColorManagement, ShaderChunk } from 'three'
import { Compile, useFrameloop } from './useCompiled'

// ShaderGradient's colours were tuned before three r152 turned ColorManagement on: with it on, its hex colours are
// converted to linear but its canvas is `linear` (no conversion back), which mutes them. Decorative only.
ColorManagement.enabled = false
// Same shim as ShaderGradientCanvas: its shaders still #include chunks three removed in r151+
Object.assign(ShaderChunk, { uv2_pars_vertex: '', uv2_vertex: '', uv2_pars_fragment: '', encodings_fragment: '' })

/**
 * Slow dark pink/indigo gradient behind the landing screen (ShaderGradient, github.com/ruucm/shadergradient, MIT).
 * Hosted in our own R3F Canvas with exactly ShaderGradientCanvas's settings (dpr, camera fov 45, linear, flat), so
 * that the frame loop can be controlled: shaders are compiled asynchronously before the first frame, and nothing
 * renders while the tab is hidden.
 * frozen=true: the current frame is copied to a still image and the WebGL canvas is unmounted, so nothing keeps
 * rendering while an FOV is analysed or viewed. lightType '3d' (the 'env' presets fetch HDRs from a third party).
 */
export default function GradientBg({ frozen, lite, active }: { frozen: boolean; lite: boolean; active: boolean }) {
  const wrap = useRef<HTMLDivElement>(null)
  const [still, setStill] = useState<string | null>(null)
  const { frameloop, compiled, setCompiled } = useFrameloop(active && !frozen)
  const ready = useCallback(() => setCompiled(true), [setCompiled])

  useEffect(() => {
    if (!frozen || still) return
    const c = wrap.current?.querySelector('canvas')
    try {
      setStill(c && c.width && compiled ? c.toDataURL('image/jpeg', 0.8) : 'none')
    } catch {
      setStill('none')
    }
  }, [frozen, still, compiled])

  useEffect(() => {
    if (!frozen) setStill(null)
  }, [frozen])

  if (frozen && still)
    return still === 'none' ? null : (
      <div className="fx-gradient fx-gradient--still" style={{ backgroundImage: `url(${still})` }} aria-hidden="true" />
    )

  return (
    <div ref={wrap} className={`fx-gradient ${compiled ? 'is-ready' : ''}`} aria-hidden="true">
      <Canvas
        frameloop={frameloop}
        dpr={lite ? 0.6 : 1}
        camera={{ fov: 45 }}
        linear
        flat
        resize={{ offsetSize: true }}
        gl={{ preserveDrawingBuffer: true, powerPreference: 'low-power', antialias: false }}
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      >
        <Compile onReady={ready} />
        <ShaderGradient
          control="props"
          type="plane"
          animate="on"
          uSpeed={0.06}
          uStrength={3.2}
          uDensity={1.5}
          uFrequency={5.5}
          uAmplitude={0}
          positionX={-1.4}
          positionY={0}
          positionZ={0}
          rotationX={0}
          rotationY={10}
          rotationZ={50}
          color1="#4B3FB3"
          color2="#F6C1D9"
          color3="#050409"
          reflection={0.08}
          brightness={0.8}
          lightType="3d"
          grain="off"
          cAzimuthAngle={180}
          cPolarAngle={90}
          cDistance={2.2}
          cameraZoom={1}
          enableTransition={false}
        />
      </Canvas>
    </div>
  )
}
