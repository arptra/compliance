import { EChart } from '../EChart'
export function CumulativeChart({ rows }: { rows: Array<{ date: string; actual: number; expected: number }> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: ['actual cumulative', 'expected cumulative'] }, xAxis: { type: 'category', data: rows.map((r) => r.date) }, yAxis: { type: 'value' }, series: [{ name: 'actual cumulative', type: 'line', data: rows.map((r) => r.actual) }, { name: 'expected cumulative', type: 'line', data: rows.map((r) => r.expected) }] }} height={280} />
}
