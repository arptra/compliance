export function ReviewSummaryBar({ summary, mode, version }: { summary: Record<string, number|string|null|undefined>; mode: string; version?: string|null }) {
  return <div className='card' style={{ marginTop: 12 }}>
    Reviewed: {Number(summary.reviewed_rows ?? 0)} | True: {Number(summary.true_count ?? 0)} | False: {Number(summary.false_count ?? 0)} | Uncertain: {Number(summary.uncertain_count ?? 0)} | Precision: {summary.precision_reviewed ?? 'n/a'} | Mode: {mode} | Active model: {version ?? 'none'}
  </div>
}
