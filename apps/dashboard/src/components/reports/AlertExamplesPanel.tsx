import type { AlertExampleCard } from '../../types/api'
import { TermHelp } from './TermHelp'

export function AlertExamplesPanel({ rows }: { rows: AlertExampleCard[] }) {
  if (!rows.length) return <div className='card'><h3>Top alert examples <TermHelp term='alert_example' /></h3><div>Примеры отключены или отсутствуют.</div></div>
  return <div className='card'><h3>Top alert examples <TermHelp term='alert_example' /></h3><div className='examples-grid'>{rows.map((r, i) => <article key={i} className='example-card'><div className='example-head'><strong>{r.category}</strong><span className={`badge ${r.priority}`}>{r.priority}</span></div><p title={r.text}>{r.text.length > 220 ? `${r.text.slice(0, 220)}…` : r.text}</p><small>{r.reason}</small></article>)}</div></div>
}
