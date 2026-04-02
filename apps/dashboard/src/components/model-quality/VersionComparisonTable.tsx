import type { ModelVersionMetricsItem } from '../../types/api'

export function VersionComparisonTable({ versions }: { versions: ModelVersionMetricsItem[] }) {
  return <div className='card'>
    <h4>Version comparison</h4>
    <table className='table'>
      <thead><tr><th>version</th><th>train rows</th><th>precision reviewed</th><th>p@50</th><th>p@100</th><th>active</th></tr></thead>
      <tbody>{versions.map((v) => <tr key={v.version_id}><td>{v.version_id}</td><td>{v.train_rows ?? '—'}</td><td>{v.precision_reviewed ?? '—'}</td><td>{v.precision_at_50 ?? '—'}</td><td>{v.precision_at_100 ?? '—'}</td><td>{v.active ? 'yes' : 'no'}</td></tr>)}</tbody>
    </table>
  </div>
}
