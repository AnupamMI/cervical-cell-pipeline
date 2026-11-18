import { useId } from 'react'
import { SoapBubbleToggle } from './soap-bubble'

// Same layout and labelling as Switch, with a soap-bubble glass thumb.
export function BubbleSwitch({ checked, onChange, label, hint, disabled }: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  hint?: string
  disabled?: boolean
}) {
  const id = useId()
  return (
    <div className="switch bswitch">
      <div className="bswitch__row">
        <SoapBubbleToggle checked={checked} onCheckedChange={onChange} disabled={disabled}
          aria-labelledby={id + '-l'} aria-describedby={hint ? id + '-h' : undefined}
          color="#3d35a8" scale={0.78} className="bswitch__toggle" />
        <span id={id + '-l'} className="switch__label" onClick={() => !disabled && onChange(!checked)}>{label}</span>
      </div>
      {hint && <span id={id + '-h'} className="switch__hint">{hint}</span>}
    </div>
  )
}

// Loaded lazily by Landing, so the WebGL code stays out of the first paint.
export { SoapBubbleButton, SoapBubbleMaterial } from './soap-bubble'
