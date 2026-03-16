import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'

export default function SettingsPage(){
  const health = useQuery({queryKey:['health'],queryFn:()=>apiGet('/api/health')})
  const tags = useQuery({queryKey:['tags'],queryFn:()=>apiGet('/api/meta/tags')})
  return <div className='card-grid'><div className='card'><h3>Health</h3><pre>{JSON.stringify(health.data,null,2)}</pre></div><div className='card'><h3>Tags</h3><pre>{JSON.stringify(tags.data,null,2)}</pre></div></div>
}
