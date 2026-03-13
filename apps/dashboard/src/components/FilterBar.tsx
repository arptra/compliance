import { useFilters } from '../state/filters'

export function FilterBar() {
  const f = useFilters()
  return <div className="filters">
    <input type="date" value={f.date_from || ''} onChange={(e)=>f.set({date_from:e.target.value || undefined})} />
    <input type="date" value={f.date_to || ''} onChange={(e)=>f.set({date_to:e.target.value || undefined})} />
    <select value={f.baseline_mode} onChange={(e)=>f.set({baseline_mode:e.target.value as any})}>
      <option value="previous_period">previous_period</option>
      <option value="same_weekday">same_weekday</option>
      <option value="seasonal">seasonal</option>
      <option value="custom_range">custom_range</option>
    </select>
    <button onClick={f.reset}>Reset filters</button>
  </div>
}
