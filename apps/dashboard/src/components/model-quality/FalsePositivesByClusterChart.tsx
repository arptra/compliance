import type { ModeMetrics } from '../../types/api'

export function FalsePositivesByClusterChart({ mode }: { mode: ModeMetrics | undefined }) {
  return <div className='card'>
    <h4>False positives by cluster</h4>
    <table className='table'><thead><tr><th>cluster</th><th>false</th></tr></thead>
      <tbody>{(mode?.by_cluster ?? []).slice(0, 10).map((r) => <tr key={r.name}><td>{r.name}</td><td>{r.false_count}</td></tr>)}</tbody></table>
  </div>
}
