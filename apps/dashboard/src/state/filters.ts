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
  categoryMode: 'top' | 'custom' | 'all'
  topN: number
  categories: string[]
  includeOther: boolean
  set: (patch: Partial<FilterState>) => void
  reset: () => void
}

const initial: Omit<FilterState, 'set' | 'reset'> = {
  baseline_mode: 'previous_period',
  metric: 'count',
  categoryMode: 'top',
  topN: 10,
  categories: [],
  includeOther: true,
}

export const useFilters = create<FilterState>((set) => ({ ...initial, set: (patch) => set(patch), reset: () => set(initial) }))
