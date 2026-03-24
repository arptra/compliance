import type { ModeMetrics } from '../../types/api'

export function ScoreBucketChart({ mode }: { mode: ModeMetrics | undefined }) {
  return <div className='card'>
    <h4>Precision by score bucket ({mode?.mode ?? 'n/a'})</h4>
    <table className='table'><thead><tr><th>bucket</th><th>reviewed</th><th>true</th><th>precision</th></tr></thead>
      <tbody>{(mode?.by_score_bucket ?? []).map((b) => <tr key={b.bucket}><td>{b.bucket}</td><td>{b.reviewed_count}</td><td>{b.true_count}</td><td>{b.precision ?? '—'}</td></tr>)}</tbody></table>
  </div>
}
