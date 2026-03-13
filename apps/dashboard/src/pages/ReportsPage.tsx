import { useMutation } from '@tanstack/react-query'
import { apiPost } from '../lib/api'

export default function ReportsPage(){
  const m = useMutation({mutationFn:()=>apiPost<{markdown?:string,html?:string}>('/api/reports/executive',{filters:{},output_format:'json'})})
  return <div><button onClick={()=>m.mutate()}>Build Executive report</button><pre>{JSON.stringify(m.data,null,2)}</pre></div>
}
