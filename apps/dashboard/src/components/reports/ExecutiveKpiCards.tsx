import { TermHelp } from './TermHelp'
import type { ExecutiveKpis } from '../../types/api'

export function ExecutiveKpiCards({ kpis }: { kpis: ExecutiveKpis }) {
  const cards = [
    { title: 'Total complaints', term: 'expected', value: kpis.total_complaints, subtitle: 'Всего жалоб за период' },
    { title: 'Delta vs baseline', term: 'delta', value: `${kpis.delta_abs >= 0 ? '+' : ''}${kpis.delta_abs} (${kpis.delta_pct?.toFixed(1) ?? 'н/д'}%)`, subtitle: 'Отклонение к baseline' },
    { title: 'Categories above baseline', term: 'categories_above_baseline', value: kpis.categories_above_baseline, subtitle: 'Категорий выше ожидаемого' },
    { title: 'Top growth category', term: 'category_contribution', value: kpis.top_growth_category ?? '—', subtitle: 'Основной вклад в рост' },
    { title: 'Pattern risk', term: 'pattern_risk', value: kpis.pattern_risk.label, subtitle: kpis.pattern_risk.score != null ? `score ${kpis.pattern_risk.score.toFixed(2)}` : 'Недоступно' },
  ]
  if (kpis.primary_area) {
    cards.push({ title: 'Primary area / owner', term: 'owner', value: kpis.primary_area.label, subtitle: kpis.primary_area.confidence_note })
  }
  return <div className='card-grid executive-kpi-grid'>{cards.map((c) => <div key={c.title} className='card'><h3>{c.title} <TermHelp term={c.term} /></h3><div className='kpi-value'>{c.value}</div><div className='kpi-subtitle'>{c.subtitle}</div></div>)}</div>
}
