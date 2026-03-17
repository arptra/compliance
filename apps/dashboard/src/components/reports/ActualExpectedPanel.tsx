import { EChart } from '../EChart'
import type { ActualExpectedPoint } from '../../types/api'
import { TermHelp } from './TermHelp'

export function ActualExpectedPanel({ rows }: { rows: ActualExpectedPoint[] }) {
  return <div className='card'><h3>Actual vs Expected <TermHelp term='expected' /></h3>{rows.length ? <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: ['Actual', 'Expected'] }, xAxis: { type: 'category', data: rows.map((r) => r.date) }, yAxis: { type: 'value' }, series: [{ name: 'Actual', type: 'line', smooth: true, data: rows.map((r) => r.actual), lineStyle: { color: '#0f766e' } }, { name: 'Expected', type: 'line', smooth: true, data: rows.map((r) => r.expected), lineStyle: { color: '#64748b' } }] }} height={300} /> : <div>Недостаточно данных</div>}</div>
}
