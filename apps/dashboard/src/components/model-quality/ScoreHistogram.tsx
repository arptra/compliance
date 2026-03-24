import type { ModeMetrics } from '../../types/api'

export function ScoreHistogram({ mode }: { mode: ModeMetrics | undefined }) {
  return <div className='card'>
    <h4>Score buckets ({mode?.mode ?? 'n/a'})</h4>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0,1fr))', gap: 6 }}>
      {(mode?.by_score_bucket ?? []).map((b) => <div key={b.bucket} style={{ border: '1px solid #e2e8f0', padding: 8 }}><div>{b.bucket}</div><div>{b.reviewed_count}</div></div>)}
    </div>
  </div>
}
