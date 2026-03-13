import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'

export default function CategoriesPage() {
  const q = useQuery({queryKey:['categories'],queryFn:()=>apiGet<{rows:Array<{category:string,count:number,share:number}>}>('/api/categories')})
  if (q.isLoading) return <div>Loading...</div>
  if (q.error) return <div>Error</div>
  return <table className='table'><thead><tr><th>Category</th><th>Count</th><th>Share</th></tr></thead><tbody>{q.data?.rows.map(r=><tr key={r.category}><td>{r.category}</td><td>{r.count}</td><td>{(r.share*100).toFixed(1)}%</td></tr>)}</tbody></table>
}
