import { useState } from 'react'
import type { CategoryPriorityRow } from '../../types/api'
import { TermHelp } from './TermHelp'

export function CategoryPriorityTable({ rows }: { rows: CategoryPriorityRow[] }) {
  const [showAll, setShowAll] = useState(false)
  const data = showAll ? rows : rows.slice(0, 7)
  return <div className='card'><h3>Categories above baseline <TermHelp term='categories_above_baseline' /></h3><table className='table'><thead><tr><th>Category</th><th>Actual</th><th>Expected</th><th>Delta <TermHelp term='delta' /></th><th>Delta % <TermHelp term='delta_pct' /></th><th>Priority <TermHelp term='priority' /></th></tr></thead><tbody>{data.map((r) => <tr key={r.category}><td>{r.category}</td><td>{r.actual.toFixed(0)}</td><td>{r.expected.toFixed(0)}</td><td>{r.delta >= 0 ? '+' : ''}{r.delta.toFixed(0)}</td><td>{r.delta_pct == null ? 'н/д' : `${r.delta_pct.toFixed(1)}%`}</td><td><span className={`badge ${r.priority}`}>{r.priority}</span></td></tr>)}</tbody></table>{rows.length > 7 && <button onClick={() => setShowAll((v) => !v)}>{showAll ? 'Show less' : 'Show more'}</button>}</div>
}
