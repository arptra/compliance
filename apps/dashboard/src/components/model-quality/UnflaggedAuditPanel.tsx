import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { apiGet, apiPost } from '../../lib/api'
import type { UnflaggedAuditSample } from '../../types/api'

export function UnflaggedAuditPanel() {
  const [sampleSize, setSampleSize] = useState(20)
  const samplesQ = useQuery({ queryKey: ['audit-samples'], queryFn: () => apiGet<UnflaggedAuditSample[]>('/api/audit/unflagged/samples') })
  const createM = useMutation({ mutationFn: () => apiPost<UnflaggedAuditSample>('/api/audit/unflagged/create', { pattern_tag: 'latest', sample_size: sampleSize }), onSuccess: () => samplesQ.refetch() })

  return <div className='card'>
    <h4>Unflagged Audit</h4>
    <div className='filters'>
      <input type='number' value={sampleSize} onChange={(e) => setSampleSize(Number(e.target.value))} />
      <button onClick={() => createM.mutate()} disabled={createM.isPending}>Create audit sample</button>
    </div>
    {!samplesQ.data?.length ? <div>No audit samples yet.</div> : <table className='table'>
      <thead><tr><th>sample</th><th>created</th><th>size</th><th>reviewed</th><th>hidden positive rate</th></tr></thead>
      <tbody>{samplesQ.data.map((s) => <tr key={s.sample_id}><td>{s.sample_id}</td><td>{s.created_at}</td><td>{s.sample_size}</td><td>{s.estimate?.reviewed_in_sample ?? 0}</td><td>{s.estimate?.estimated_hidden_positive_rate ?? '—'}</td></tr>)}</tbody>
    </table>}
  </div>
}
