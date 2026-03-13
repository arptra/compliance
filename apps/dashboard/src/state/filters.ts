import { create } from 'zustand'

type FilterState = {
  date_from?: string
  date_to?: string
  baseline_mode: 'previous_period' | 'same_weekday' | 'seasonal' | 'custom_range'
  baseline_date_from?: string
  baseline_date_to?: string
  viz_tag?: string
  pattern_tag?: string
  metric: 'count' | 'share' | 'delta' | 'anomaly' | 'pattern'
  set: (patch: Partial<FilterState>) => void
  reset: () => void
}

const initial = { baseline_mode: 'previous_period', metric: 'count' } as const

export const useFilters = create<FilterState>((set) => ({ ...initial, set: (patch) => set(patch), reset: () => set(initial) }))
