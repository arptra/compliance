import type { ReviewDatasetFiltersState } from '../../types/api'

type Props = {
  filters: ReviewDatasetFiltersState
  onChange: (patch: Partial<ReviewDatasetFiltersState>) => void
}

export function ReviewDatasetFilters({ filters, onChange }: Props) {
  return <div className='filters'>
    <input placeholder='q: row_id/comment' value={filters.q ?? ''} onChange={(e) => onChange({ q: e.target.value })} />
    <input type='date' value={filters.date_from ?? ''} onChange={(e) => onChange({ date_from: e.target.value })} />
    <input type='date' value={filters.date_to ?? ''} onChange={(e) => onChange({ date_to: e.target.value })} />
    <input placeholder='reviewer' value={filters.reviewer ?? ''} onChange={(e) => onChange({ reviewer: e.target.value })} />
    <input placeholder='pattern_tag' value={filters.pattern_tag ?? ''} onChange={(e) => onChange({ pattern_tag: e.target.value })} />
    <select value={filters.verdict ?? ''} onChange={(e) => onChange({ verdict: e.target.value || undefined })}>
      <option value=''>all verdicts</option>
      <option value='true'>true</option>
      <option value='false'>false</option>
      <option value='uncertain'>uncertain</option>
    </select>
    <input placeholder='category' value={filters.category ?? ''} onChange={(e) => onChange({ category: e.target.value })} />
    <input placeholder='reason_code' value={filters.reason_code ?? ''} onChange={(e) => onChange({ reason_code: e.target.value })} />
    <input placeholder='model_version' value={filters.model_version ?? ''} onChange={(e) => onChange({ model_version: e.target.value })} />
  </div>
}
