import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
export default function PatternMonitorPage(){ const q=useQuery({queryKey:['pm'],queryFn:()=>apiGet('/api/pattern-monitor/summary?pattern_tag=latest')}); return <pre>{JSON.stringify(q.data,null,2)}</pre> }
