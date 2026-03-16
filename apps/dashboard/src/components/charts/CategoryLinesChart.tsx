import { EChart } from '../EChart'
export function CategoryLinesChart({ dates, categories, matrix }: { dates: string[]; categories: string[]; matrix: Record<string, number[]> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: categories }, xAxis: { type: 'category', data: dates }, yAxis: { type: 'value' }, series: categories.map((c) => ({ name: c, type: 'line', data: matrix[c] ?? [] })) }} height={340} />
}
