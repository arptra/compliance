import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import { useFilters } from '../state/filters'
import { useShallow } from 'zustand/react/shallow'
import { ActualExpectedChart } from '../components/charts/ActualExpectedChart'
import { DailyDeltaBars } from '../components/charts/DailyDeltaBars'
import { CumulativeChart } from '../components/charts/CumulativeChart'
import { CategoryStackedArea } from '../components/charts/CategoryStackedArea'
import { CategoryShareArea } from '../components/charts/CategoryShareArea'
import { CategoryLinesChart } from '../components/charts/CategoryLinesChart'
import { WeekdayHourHeatmap } from '../components/charts/WeekdayHourHeatmap'
import { CalendarHeatmap } from '../components/charts/CalendarHeatmap'
import { ContributionChart } from '../components/charts/ContributionChart'

type CategoryRow = { date: string; category: string; count: number; share: number }

export default function TimeseriesPage() {
  const f = useFilters(useShallow((s) => ({
    date_from: s.date_from,
    date_to: s.date_to,
    baseline_mode: s.baseline_mode,
    baseline_date_from: s.baseline_date_from,
    baseline_date_to: s.baseline_date_to,
    categoryMode: s.categoryMode,
    topN: s.topN,
    includeOther: s.includeOther,
    categories: s.categories,
  })))
  const [granularity, setGranularity] = useState<'D' | 'W' | 'M'>('D')
  const [categoryChartMode, setCategoryChartMode] = useState<'stacked' | 'lines'>('stacked')

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
  }, [
    granularity,
    f.date_from,
    f.date_to,
    f.baseline_mode,
    f.baseline_date_from,
    f.baseline_date_to,
    f.categoryMode,
    f.topN,
    f.includeOther,
    f.categories,
  ])

  const queryCommon = { staleTime: 30_000, refetchOnWindowFocus: false, placeholderData: (prev: unknown) => prev }
  const overall = useQuery({ queryKey: ['ts-overall-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/overall?${qs}`), ...queryCommon })
  const byCategory = useQuery({ queryKey: ['ts-by-category-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/by-category?${qs}`), ...queryCommon })
  const heatmap = useQuery({ queryKey: ['ts-heatmap-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/heatmap?${qs}`), ...queryCommon })
  const compare = useQuery({ queryKey: ['ts-compare-v2', qs], queryFn: () => apiGet<any>(`/api/timeseries/compare?${qs}`), ...queryCommon })

  if (overall.isLoading || byCategory.isLoading || heatmap.isLoading || compare.isLoading) return <div className='card'>Загрузка timeseries...</div>
  if (overall.error) return <div className='card'>Ошибка overall: {(overall.error as Error).message}</div>

  const delta = overall.data?.delta ?? []
  const cumulative = overall.data?.cumulative ?? []
  const summary = overall.data?.summary ?? {}

  const rows: CategoryRow[] = byCategory.data?.rows ?? []

  const chartData = useMemo(() => {
    if (!rows.length) {
      return { categories: [] as string[], dates: [] as string[], countMatrix: {} as Record<string, number[]>, shareMatrix: {} as Record<string, number[]> }
    }

    const categories: string[] = []
    const dates: string[] = []
    const catIdx = new Map<string, number>()
    const dateIdx = new Map<string, number>()

    for (const r of rows) {
      if (!catIdx.has(r.category)) {
        catIdx.set(r.category, categories.length)
        categories.push(r.category)
      }
      if (!dateIdx.has(r.date)) {
        dateIdx.set(r.date, dates.length)
        dates.push(r.date)
      }
    }

    dates.sort()
    dateIdx.clear()
    dates.forEach((d, i) => dateIdx.set(d, i))

    const countMatrix: Record<string, number[]> = {}
    const shareMatrix: Record<string, number[]> = {}
    for (const c of categories) {
      countMatrix[c] = new Array(dates.length).fill(0)
      shareMatrix[c] = new Array(dates.length).fill(0)
    }

    for (const r of rows) {
      const di = dateIdx.get(r.date)
      if (di === undefined) continue
      countMatrix[r.category][di] += r.count
      shareMatrix[r.category][di] += r.share
    }

    return { categories, dates, countMatrix, shareMatrix }
  }, [rows])

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
      {chartData.dates.length === 0 ? <div>Нет данных по категориям</div> : categoryChartMode === 'stacked' ? <CategoryStackedArea dates={chartData.dates} categories={chartData.categories} matrix={chartData.countMatrix} /> : <CategoryLinesChart dates={chartData.dates} categories={chartData.categories} matrix={chartData.countMatrix} />}
    </div>

    <div className='card' style={{ marginTop: 12 }}><h3>100% shares by category</h3>{chartData.dates.length ? <CategoryShareArea dates={chartData.dates} categories={chartData.categories} matrix={chartData.shareMatrix} /> : <div>Нет данных</div>}</div>

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
