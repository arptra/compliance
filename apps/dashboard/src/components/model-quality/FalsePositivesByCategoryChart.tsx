import type { ModeMetrics } from '../../types/api'

export function FalsePositivesByCategoryChart({ mode }: { mode: ModeMetrics | undefined }) {
  return <div className='card'>
    <h4>False positives by category</h4>
    <table className='table'><thead><tr><th>category</th><th>false</th></tr></thead>
      <tbody>{(mode?.by_category ?? []).slice(0, 10).map((r) => <tr key={r.name}><td>{r.label_ru ?? r.name}</td><td>{r.false_count}</td></tr>)}</tbody></table>
  </div>
}
