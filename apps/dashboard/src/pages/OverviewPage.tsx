import { useOverview } from '../hooks/useOverview'

export default function OverviewPage() {
  const q = useOverview()
  if (q.isLoading) return <div>Loading...</div>
  if (q.error) return <div>Failed: {(q.error as Error).message}</div>
  const data = q.data!
  return <div>
    <div className='card-grid'>{data.kpis.map(k => <div className='card' key={k.key}><div>{k.key}</div><strong>{k.value}</strong></div>)}</div>
    <div className='card' style={{marginTop:12}}><h3>Executive summary</h3><p>{data.executive_summary}</p></div>
  </div>
}
