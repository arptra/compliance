import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import { overviewSchema } from '../types/api'
import { useFilters } from '../state/filters'

export function useOverview() {
  const f = useFilters()
  const qs = new URLSearchParams()
  if (f.date_from) qs.set('date_from', f.date_from)
  if (f.date_to) qs.set('date_to', f.date_to)
  if (f.viz_tag) qs.set('viz_tag', f.viz_tag)
  qs.set('baseline_mode', f.baseline_mode)
  qs.set('metric', f.metric)
  return useQuery({
    queryKey: ['overview', f],
    queryFn: async () => overviewSchema.parse(await apiGet(`/api/overview?${qs.toString()}`))
  })
}
