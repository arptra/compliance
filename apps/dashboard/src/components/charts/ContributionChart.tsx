import { EChart } from '../EChart'
export function ContributionChart({ rows }: { rows: Array<{ category: string; delta_abs: number }> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, xAxis: { type: 'value' }, yAxis: { type: 'category', data: rows.map((r) => r.category) }, series: [{ type: 'bar', data: rows.map((r) => r.delta_abs) }] }} height={360} />
}
