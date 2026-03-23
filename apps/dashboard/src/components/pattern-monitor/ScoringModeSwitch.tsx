import type { ChangeEvent } from 'react'

export function ScoringModeSwitch({ value, disabledModes, onChange }: { value: 'base'|'calibrated'|'reranked'; disabledModes?: string[]; onChange: (v:'base'|'calibrated'|'reranked')=>void }) {
  const disabled = new Set(disabledModes ?? [])
  return <select value={value} onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value as 'base'|'calibrated'|'reranked')}>
    <option value='base'>Base</option>
    <option value='calibrated' disabled={disabled.has('calibrated')}>Calibrated</option>
    <option value='reranked' disabled={disabled.has('reranked')}>Reranked</option>
  </select>
}
