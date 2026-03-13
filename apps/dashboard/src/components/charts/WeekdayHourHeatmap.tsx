import { EChart } from '../EChart'
export function WeekdayHourHeatmap({ rows }: { rows: Array<{ dow: number; hour: number; value: number }> }) {
  return <EChart option={{ tooltip: {}, xAxis: { type: 'category', data: Array.from({ length: 24 }, (_, i) => String(i)) }, yAxis: { type: 'category', data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] }, visualMap: { min: 0, max: Math.max(1, ...rows.map((r) => r.value)), orient: 'horizontal' }, series: [{ type: 'heatmap', data: rows.map((r) => [r.hour, r.dow, r.value]) }] }} height={300} />
}
