import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
export default function PatternFitPage(){ const q=useQuery({queryKey:['pf'],queryFn:()=>apiGet('/api/pattern-fit/summary?tag=latest')}); return <pre>{JSON.stringify(q.data,null,2)}</pre> }
