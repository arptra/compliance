import { EChart } from '../EChart'
export function CategoryStackedArea({ dates, categories, matrix }: { dates: string[]; categories: string[]; matrix: Record<string, number[]> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: categories }, xAxis: { type: 'category', data: dates }, yAxis: { type: 'value' }, series: categories.map((c) => ({ name: c, type: 'line', stack: 't', areaStyle: {}, data: matrix[c] ?? [] })) }} height={360} />
}
