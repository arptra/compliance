import { EChart } from '../EChart'
export function DailyDeltaBars({ rows }: { rows: Array<{ date: string; delta_abs: number }> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, xAxis: { type: 'category', data: rows.map((r) => r.date) }, yAxis: { type: 'value' }, series: [{ type: 'bar', data: rows.map((r) => r.delta_abs), itemStyle: { color: (p: any) => (p.value >= 0 ? '#dc2626' : '#2563eb') } }] }} height={260} />
}
