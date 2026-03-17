import { EChart } from '../EChart'
import type { ContributionRow } from '../../types/api'
import { TermHelp } from './TermHelp'

export function ContributionPanel({ rows }: { rows: ContributionRow[] }) {
  return <div className='card'><h3>Category contribution to growth <TermHelp term='category_contribution' /></h3>{rows.length ? <EChart option={{ tooltip: { trigger: 'axis' }, xAxis: { type: 'value' }, yAxis: { type: 'category', data: rows.map((r) => r.category) }, series: [{ type: 'bar', data: rows.map((r) => r.delta), itemStyle: { color: '#2563eb' } }] }} height={300} /> : <div>Нет выраженного роста категорий</div>}</div>
}
