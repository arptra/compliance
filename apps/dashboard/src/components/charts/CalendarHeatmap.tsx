import { EChart } from '../EChart'
export function CalendarHeatmap({ rows }: { rows: Array<{ date: string; value: number }> }) {
  return <EChart option={{ tooltip: {}, xAxis: { type: 'category', data: rows.map((r) => r.date) }, yAxis: { type: 'value' }, series: [{ type: 'bar', data: rows.map((r) => r.value) }] }} height={220} />
}
