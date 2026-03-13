import { EChart } from '../EChart'
export function CategoryShareArea({ dates, categories, matrix }: { dates: string[]; categories: string[]; matrix: Record<string, number[]> }) {
  return <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: categories }, xAxis: { type: 'category', data: dates }, yAxis: { type: 'value', max: 1 }, series: categories.map((c) => ({ name: c, type: 'line', stack: 's', areaStyle: {}, data: matrix[c] ?? [] })) }} height={360} />
}
