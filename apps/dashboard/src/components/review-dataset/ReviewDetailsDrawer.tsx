import { Link } from 'react-router-dom'
import type { FeedbackDatasetItem } from '../../types/api'

type Props = {
  item: FeedbackDatasetItem | null
  onClose: () => void
}

export function ReviewDetailsDrawer({ item, onClose }: Props) {
  if (!item) return null
  return <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.45)', display: 'flex', justifyContent: 'flex-end', zIndex: 50 }} onClick={onClose}>
    <div className='card' style={{ width: 'min(680px, 92vw)', height: '100vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}>
      <h3>Review details</h3>
      <p><b>row_id:</b> {item.row_id}</p>
      <p><b>pattern_tag:</b> {item.pattern_tag ?? '—'}</p>
      <p><b>scores:</b> base={item.base_score ?? '—'} rerank={item.rerank_score ?? '—'}</p>
      <p><b>verdict:</b> {item.verdict}</p>
      <p><b>reason:</b> {item.reason_code ?? '—'}</p>
      <p><b>comment:</b> {item.comment ?? '—'}</p>
      <p><b>cluster:</b> {item.subcategory ?? '—'}</p>
      <p><b>text:</b></p>
      <pre style={{ whiteSpace: 'pre-wrap' }}>{item.comment || 'Текст жалобы недоступен в feedback-таблице.'}</pre>
      <div style={{ display: 'flex', gap: 8 }}>
        <Link to={`/pattern-monitor?row_id=${encodeURIComponent(item.row_id)}`}><button>Open in Pattern Monitor</button></Link>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  </div>
}
