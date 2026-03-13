import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
export default function TimeseriesPage(){ const q=useQuery({queryKey:['ts'],queryFn:()=>apiGet<Array<{date:string,actual:number}>>('/api/timeseries/overall')}); if(q.isLoading)return <div>Loading...</div>; return <pre>{JSON.stringify(q.data?.slice(0,20),null,2)}</pre> }
