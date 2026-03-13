import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import { useFilters } from '../state/filters'
import { ActualExpectedChart } from '../components/charts/ActualExpectedChart'
import { DailyDeltaBars } from '../components/charts/DailyDeltaBars'
import { CumulativeChart } from '../components/charts/CumulativeChart'
import { CategoryStackedArea } from '../components/charts/CategoryStackedArea'
import { CategoryShareArea } from '../components/charts/CategoryShareArea'
import { CategoryLinesChart } from '../components/charts/CategoryLinesChart'
import { WeekdayHourHeatmap } from '../components/charts/WeekdayHourHeatmap'
import { CalendarHeatmap } from '../components/charts/CalendarHeatmap'
import { ContributionChart } from '../components/charts/ContributionChart'

export default function TimeseriesPage() {
  const f = useFilters()
  const [granularity, setGranularity] = useState<'D'|'W'|'M'>('D')
  const [categoryChartMode, setCategoryChartMode] = useState<'stacked'|'lines'>('stacked')

  const qs = useMemo(() => {
    const q = new URLSearchParams()
    if (f.date_from) q.set('date_from', f.date_from)
    if (f.date_to) q.set('date_to', f.date_to)
    q.set('granularity', granularity)
    q.set('baseline_mode', f.baseline_mode)
    if (f.baseline_date_from) q.set('baseline_date_from', f.baseline_date_from)
    if (f.baseline_date_to) q.set('baseline_date_to', f.baseline_date_to)
    q.set('category_mode', f.categoryMode)
    q.set('top_n', String(f.topN))
    q.set('include_other', String(f.includeOther))
    for (const c of f.categories) q.append('categories', c)
    return q.toString()
  }, [f, granularity])

  const overall = useQuery({ queryKey: ['ts-overall-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/overall?${qs}`) })
  const byCategory = useQuery({ queryKey: ['ts-by-category-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/by-category?${qs}`) })
  const heatmap = useQuery({ queryKey: ['ts-heatmap-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/heatmap?${qs}`) })
  const compare = useQuery({ queryKey: ['ts-compare-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/compare?${qs}`) })

  if (overall.isLoading || byCategory.isLoading || heatmap.isLoading || compare.isLoading) return <div className='card'>Загрузка timeseries...</div>
  if (overall.error) return <div className='card'>Ошибка overall: {(overall.error as Error).message}</div>

  const actual = overall.data?.actual ?? []
  const expected = overall.data?.expected ?? []
  const delta = overall.data?.delta ?? []
  const cumulative = overall.data?.cumulative ?? []
  const summary = overall.data?.summary ?? {}

  const rows: Array<{date:string;category:string;count:number;share:number}> = byCategory.data?.rows ?? []
  const categories: string[] = Array.from(new Set(rows.map((r) => String(r.category))))
  const dates: string[] = Array.from(new Set(rows.map((r) => String(r.date)))).sort()
  const countMatrix: Record<string, number[]> = {}
  const shareMatrix: Record<string, number[]> = {}
  for (const c of categories) {
    countMatrix[c] = dates.map((d) => rows.filter((r) => r.date === d && r.category === c).reduce((s, r) => s + r.count, 0))
    shareMatrix[c] = dates.map((d) => rows.filter((r) => r.date === d && r.category === c).reduce((s, r) => s + r.share, 0))
  }

  const mergedAE = delta.map((d: any) => ({ date: d.date, actual: d.actual, expected: d.expected, delta_abs: d.delta_abs, delta_pct: d.delta_pct }))

  return <div>
    <div className='card'>
      <h3>Timeseries controls</h3>
      <div className='filters'>
        <label>Granularity <select value={granularity} onChange={(e) => setGranularity(e.target.value as any)}><option value='D'>day</option><option value='W'>week</option><option value='M'>month</option></select></label>
        <label>Category chart <select value={categoryChartMode} onChange={(e) => setCategoryChartMode(e.target.value as any)}><option value='stacked'>stacked</option><option value='lines'>lines</option></select></label>
      </div>
    </div>

    <div className='card-grid' style={{ marginTop: 12 }}>
      <div className='card'><div>Total complaints</div><strong>{(summary.actual_total ?? 0).toFixed(0)}</strong></div>
      <div className='card'><div>Expected complaints</div><strong>{(summary.expected_total ?? 0).toFixed(0)}</strong></div>
      <div className='card'><div>Delta abs</div><strong>{(summary.delta_abs ?? 0).toFixed(0)}</strong></div>
    </div>

    <div className='card' style={{ marginTop: 12 }}><h3>Actual vs Expected</h3>{mergedAE.length ? <ActualExpectedChart rows={mergedAE} /> : <div>Нет данных</div>}</div>
    <div className='card' style={{ marginTop: 12 }}><h3>Daily Excess / Delta</h3>{delta.length ? <DailyDeltaBars rows={delta} /> : <div>Нет данных</div>}</div>
    <div className='card' style={{ marginTop: 12 }}><h3>Cumulative complaints</h3>{cumulative.length ? <CumulativeChart rows={cumulative} /> : <div>Нет данных</div>}</div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Category structure over time</h3>
      {dates.length === 0 ? <div>Нет данных по категориям</div> : categoryChartMode === 'stacked' ? <CategoryStackedArea dates={dates} categories={categories} matrix={countMatrix} /> : <CategoryLinesChart dates={dates} categories={categories} matrix={countMatrix} />}
    </div>

    <div className='card' style={{ marginTop: 12 }}><h3>100% shares by category</h3>{dates.length ? <CategoryShareArea dates={dates} categories={categories} matrix={shareMatrix} /> : <div>Нет данных</div>}</div>

    <div className='card' style={{ marginTop: 12 }}><h3>Weekday × hour heatmap</h3>{(heatmap.data?.weekday_hour ?? []).length ? <WeekdayHourHeatmap rows={heatmap.data.weekday_hour} /> : <div>Нет часовой детализации</div>}</div>
    <div className='card' style={{ marginTop: 12 }}><h3>Calendar heatmap</h3>{(heatmap.data?.calendar ?? []).length ? <CalendarHeatmap rows={heatmap.data.calendar} /> : <div>Нет календарных данных</div>}</div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Compare to baseline</h3>
      <div>Actual: {(compare.data?.summary?.actual_total ?? 0).toFixed(0)} | Baseline: {(compare.data?.summary?.baseline_total ?? 0).toFixed(0)} | Δ: {(compare.data?.summary?.delta_abs ?? 0).toFixed(0)}</div>
      <ContributionChart rows={(compare.data?.contributions ?? []).slice(0, 15)} />
      <table className='table' style={{ marginTop: 12 }}>
        <thead><tr><th>category</th><th>actual_count</th><th>expected_count</th><th>delta_abs</th><th>delta_pct</th><th>share</th><th>contribution_to_growth</th><th>anomaly_score</th></tr></thead>
        <tbody>
          {(compare.data?.contributions ?? []).map((r: any) => (
            <tr key={r.category}><td>{r.category}</td><td>{r.actual_count.toFixed(0)}</td><td>{r.expected_count.toFixed(0)}</td><td>{r.delta_abs.toFixed(0)}</td><td>{r.delta_pct == null ? '—' : `${(r.delta_pct*100).toFixed(1)}%`}</td><td>{(r.share*100).toFixed(1)}%</td><td>{(r.contribution_to_growth*100).toFixed(1)}%</td><td>{r.anomaly_score.toFixed(2)}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
}
