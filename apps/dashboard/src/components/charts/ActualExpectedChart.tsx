import { EChart } from '../EChart'
export function ActualExpectedChart({ rows }: { rows: Array<{ date: string; actual: number; expected: number; delta_abs: number; delta_pct: number | null }> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: ['actual', 'expected'] }, xAxis: { type: 'category', data: rows.map((r) => r.date) }, yAxis: { type: 'value' }, series: [{ name: 'actual', type: 'line', data: rows.map((r) => r.actual), smooth: true }, { name: 'expected', type: 'line', data: rows.map((r) => r.expected), smooth: true }] }} height={320} />
}
