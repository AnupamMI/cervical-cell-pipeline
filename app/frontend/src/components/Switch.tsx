import { useId } from 'react'

export function Switch({ checked, onChange, label, hint, disabled, compact }: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  hint?: string
  disabled?: boolean
  compact?: boolean
}) {
  const id = useId()
  return (
    <div className={`switch ${compact ? 'switch--compact' : ''}`}>
      <button id={id} type="button" role="switch" aria-checked={checked} disabled={disabled}
        aria-describedby={hint ? id + '-h' : undefined} className="switch__btn" onClick={() => onChange(!checked)}>
        <span className="switch__track" aria-hidden="true"><span className="switch__thumb" /></span>
        <span className="switch__label">{label}</span>
      </button>
      {hint && <span id={id + '-h'} className="switch__hint">{hint}</span>}
    </div>
  )
}
