import type { ModeMetrics } from '../../types/api'

export function ReviewFunnel({ metrics }: { metrics: ModeMetrics | undefined }) {
  if (!metrics) return <div className='card'>No data</div>
  return <div className='card'>
    <h4>Review Funnel ({metrics.mode})</h4>
    <ul>
      <li>reviewed: {metrics.reviewed_rows}</li>
      <li>true: {metrics.true_count}</li>
      <li>false: {metrics.false_count}</li>
      <li>uncertain: {metrics.uncertain_count}</li>
    </ul>
  </div>
}
