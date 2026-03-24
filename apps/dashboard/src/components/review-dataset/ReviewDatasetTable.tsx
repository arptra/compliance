import type { FeedbackDatasetItem } from '../../types/api'

type Props = {
  rows: FeedbackDatasetItem[]
  onSelect: (item: FeedbackDatasetItem) => void
}

export function ReviewDatasetTable({ rows, onSelect }: Props) {
  return <div style={{ overflow: 'auto', maxHeight: '60vh' }}>
    <table className='table'>
      <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
        <tr>
          <th>review_date</th><th>row_id</th><th>pattern_tag</th><th>category</th><th>subcategory</th><th>base</th><th>rerank</th><th>verdict</th><th>reason</th><th>reviewer</th><th>model</th><th>comment</th><th>actions</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => <tr key={`${r.row_id}-${r.pattern_tag ?? 'na'}`}>
          <td>{r.review_date ?? ''}</td><td>{r.row_id}</td><td>{r.pattern_tag ?? ''}</td><td>{r.category ?? ''}</td><td>{r.subcategory ?? ''}</td><td>{r.base_score ?? ''}</td><td>{r.rerank_score ?? ''}</td><td>{r.verdict}</td><td>{r.reason_code ?? ''}</td><td>{r.reviewer ?? ''}</td><td>{r.model_version ?? ''}</td><td>{r.comment ?? ''}</td>
          <td><button onClick={() => onSelect(r)}>Details</button></td>
        </tr>)}
      </tbody>
    </table>
  </div>
}
