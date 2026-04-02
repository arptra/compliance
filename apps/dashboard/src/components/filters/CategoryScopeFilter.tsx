import { useTaxonomyLabels } from '../../hooks/useTaxonomyLabels'

export type CategoryScopeValue = {
  categoryMode: 'top' | 'custom' | 'all'
  topN: number
  categories: string[]
  includeOther: boolean
}

type Props = {
  availableCategories: string[]
  value: CategoryScopeValue
  onChange: (patch: Partial<CategoryScopeValue>) => void
}

export function CategoryScopeFilter({ availableCategories, value, onChange }: Props) {
  const labels = useTaxonomyLabels()
  return <div className='category-scope'>
    <div>
      <button onClick={() => onChange({ categoryMode: 'top' })} style={{ fontWeight: value.categoryMode === 'top' ? 700 : 400 }}>Top N</button>
      <button onClick={() => onChange({ categoryMode: 'custom' })} style={{ fontWeight: value.categoryMode === 'custom' ? 700 : 400 }}>Custom</button>
      <button onClick={() => onChange({ categoryMode: 'all' })} style={{ fontWeight: value.categoryMode === 'all' ? 700 : 400 }}>All</button>
    </div>

    {value.categoryMode === 'top' && (
      <label>Top N:
        <select value={value.topN} onChange={(e) => onChange({ topN: Number(e.target.value) })}>
          {[5, 10, 15, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
    )}

    {value.categoryMode === 'custom' && (
      <select multiple value={value.categories} onChange={(e) => onChange({ categories: Array.from(e.target.selectedOptions).map((o) => o.value) })}>
        {availableCategories.map((c) => <option key={c} value={c}>{labels.categoryLabel(c)}</option>)}
      </select>
    )}

    <label><input type='checkbox' checked={value.includeOther} onChange={(e) => onChange({ includeOther: e.target.checked })} /> include OTHER</label>
    <button onClick={() => onChange({ categoryMode: 'top', topN: 10, categories: [], includeOther: true })}>Reset</button>
  </div>
}
