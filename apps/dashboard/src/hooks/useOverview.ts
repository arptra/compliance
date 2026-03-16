import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import { overviewSchema } from '../types/api'
import { useFilters } from '../state/filters'
import { useShallow } from 'zustand/react/shallow'

export function useOverview() {
  const f = useFilters(useShallow((s) => ({
    date_from: s.date_from,
    date_to: s.date_to,
    viz_tag: s.viz_tag,
    baseline_mode: s.baseline_mode,
    baseline_date_from: s.baseline_date_from,
    baseline_date_to: s.baseline_date_to,
    metric: s.metric,
    categoryMode: s.categoryMode,
    topN: s.topN,
    includeOther: s.includeOther,
    categories: s.categories,
  })))
  const qs = new URLSearchParams()
  if (f.date_from) qs.set('date_from', f.date_from)
  if (f.date_to) qs.set('date_to', f.date_to)
  if (f.viz_tag) qs.set('viz_tag', f.viz_tag)
  qs.set('baseline_mode', f.baseline_mode)
  if (f.baseline_date_from) qs.set('baseline_date_from', f.baseline_date_from)
  if (f.baseline_date_to) qs.set('baseline_date_to', f.baseline_date_to)
  qs.set('metric', f.metric)
  qs.set('category_mode', f.categoryMode)
  qs.set('top_n', String(f.topN))
  qs.set('include_other', String(f.includeOther))
  for (const c of f.categories) qs.append('category', c)
  return useQuery({
    queryKey: ['overview', qs.toString()],
    queryFn: async () => overviewSchema.parse(await apiGet(`/api/overview?${qs.toString()}`)),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
}
