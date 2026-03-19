export function FeedbackMetricsPanel({ summary }: { summary: Record<string, unknown> | undefined }) {
  if (!summary) return null
  return <div className='card' style={{ marginTop: 12 }}>
    <h3>Model quality</h3>
    <div>Precision reviewed: {String(summary.precision_reviewed ?? 'n/a')}</div>
    <div>Precision@50: {String(summary.precision_at_50 ?? 'n/a')}</div>
    <div>Precision@100: {String(summary.precision_at_100 ?? 'n/a')}</div>
  </div>
}
