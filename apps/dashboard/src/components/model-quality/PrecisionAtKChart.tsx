import type { ModeMetrics } from '../../types/api'

export function PrecisionAtKChart({ modes }: { modes: ModeMetrics[] }) {
  return <div className='card'>
    <h4>Precision@K compare</h4>
    <table className='table'>
      <thead><tr><th>mode</th><th>@10</th><th>@20</th><th>@50</th><th>@100</th></tr></thead>
      <tbody>{modes.map((m) => <tr key={m.mode}><td>{m.mode}</td>{m.precision_at.map((p) => <td key={p.k}>{p.precision ?? '—'}</td>)}</tr>)}</tbody>
    </table>
  </div>
}
