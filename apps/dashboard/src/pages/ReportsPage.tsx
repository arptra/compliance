import { useMutation } from '@tanstack/react-query'
import { apiPost } from '../lib/api'
import { useFilters } from '../state/filters'

export default function ReportsPage(){
  const f = useFilters()
  const m = useMutation({mutationFn:()=>apiPost<{markdown?:string,html?:string}>('/api/reports/executive',{filters:{date_from:f.date_from,date_to:f.date_to,category_mode:f.categoryMode,top_n:f.topN,categories:f.categories,include_other:f.includeOther,baseline_mode:f.baseline_mode,baseline_date_from:f.baseline_date_from,baseline_date_to:f.baseline_date_to},output_format:'json'})})
  return <div><button onClick={()=>m.mutate()}>Build Executive report</button><pre>{JSON.stringify(m.data,null,2)}</pre></div>
}
