import { useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useFilters } from '../state/filters'
import { apiGet } from '../lib/api'
import { CategoryScopeFilter } from './filters/CategoryScopeFilter'

export function FilterBar() {
  const f = useFilters()
  const loc = useLocation()
  const nav = useNavigate()
  const catsQ = useQuery({ queryKey: ['cats-for-filter', f.date_from, f.date_to], queryFn: () => apiGet<{ rows: Array<{ category: string }> }>(`/api/categories?${new URLSearchParams({ date_from: f.date_from || '', date_to: f.date_to || '' }).toString()}`) })

  useEffect(() => {
    const q = new URLSearchParams(loc.search)
    const categoryMode = q.get('categoryMode') as 'top' | 'custom' | 'all' | null
    const topN = q.get('topN')
    const categories = q.get('categories')
    const includeOther = q.get('includeOther')
    f.set({
      date_from: q.get('date_from') || undefined,
      date_to: q.get('date_to') || undefined,
      categoryMode: categoryMode || f.categoryMode,
      topN: topN ? Number(topN) : f.topN,
      categories: categories ? categories.split(',').filter(Boolean) : f.categories,
      includeOther: includeOther ? includeOther === 'true' : f.includeOther,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const q = new URLSearchParams(loc.search)
    if (f.date_from) q.set('date_from', f.date_from); else q.delete('date_from')
    if (f.date_to) q.set('date_to', f.date_to); else q.delete('date_to')
    q.set('categoryMode', f.categoryMode)
    q.set('topN', String(f.topN))
    q.set('includeOther', String(f.includeOther))
    if (f.categories.length) q.set('categories', f.categories.join(',')); else q.delete('categories')
    nav({ pathname: loc.pathname, search: q.toString() }, { replace: true })
  }, [f.date_from, f.date_to, f.categoryMode, f.topN, f.categories, f.includeOther, loc.pathname])

  const available = Array.from(new Set((catsQ.data?.rows ?? []).map((r) => r.category))).filter(Boolean)

  return <div className="filters">
    <input type="date" value={f.date_from || ''} onChange={(e)=>f.set({date_from:e.target.value || undefined})} />
    <input type="date" value={f.date_to || ''} onChange={(e)=>f.set({date_to:e.target.value || undefined})} />
    <select value={f.baseline_mode} onChange={(e)=>f.set({baseline_mode:e.target.value as any})}>
      <option value="previous_period">previous_period</option>
      <option value="same_weekday">same_weekday</option>
      <option value="seasonal">seasonal</option>
      <option value="custom_range">custom_range</option>
    </select>
    {f.baseline_mode === 'custom_range' && (
      <>
        <input type="date" value={f.baseline_date_from || ''} onChange={(e)=>f.set({baseline_date_from:e.target.value || undefined})} />
        <input type="date" value={f.baseline_date_to || ''} onChange={(e)=>f.set({baseline_date_to:e.target.value || undefined})} />
      </>
    )}
    <CategoryScopeFilter
      availableCategories={available}
      value={{ categoryMode: f.categoryMode, topN: f.topN, categories: f.categories, includeOther: f.includeOther }}
      onChange={(patch) => f.set(patch as any)}
    />
    <button onClick={f.reset}>Reset filters</button>
  </div>
}
