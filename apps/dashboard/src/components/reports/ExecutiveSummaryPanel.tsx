import type { SummaryBlock } from '../../types/api'

export function ExecutiveSummaryPanel({ summary }: { summary: SummaryBlock }) {
  return <div className='card'><h3>{summary.headline}</h3><ul>{summary.bullets.map((b, i) => <li key={i}>{b}</li>)}</ul><h4>Рекомендуемые действия</h4><ul>{summary.recommended_actions.map((a, i) => <li key={i}>{a}</li>)}</ul></div>
}
