import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import type { ModelQualityResponse } from '../types/api'
import { ReviewFunnel } from '../components/model-quality/ReviewFunnel'
import { PrecisionAtKChart } from '../components/model-quality/PrecisionAtKChart'
import { ScoreHistogram } from '../components/model-quality/ScoreHistogram'
import { ScoreBucketChart } from '../components/model-quality/ScoreBucketChart'
import { FalsePositivesByCategoryChart } from '../components/model-quality/FalsePositivesByCategoryChart'
import { FalsePositivesByClusterChart } from '../components/model-quality/FalsePositivesByClusterChart'
import { VersionComparisonTable } from '../components/model-quality/VersionComparisonTable'
import { UnflaggedAuditPanel } from '../components/model-quality/UnflaggedAuditPanel'

export default function ModelQualityPage() {
  const [patternTag, setPatternTag] = useState('latest')
  const [includeUncertain, setIncludeUncertain] = useState<'ignore'|'false'|'separate'>('ignore')
  const qs = useMemo(() => new URLSearchParams({ pattern_tag: patternTag, include_uncertain_as: includeUncertain }).toString(), [patternTag, includeUncertain])
  const qualityQ = useQuery({ queryKey: ['model-quality', qs], queryFn: () => apiGet<ModelQualityResponse>(`/api/model-quality?${qs}`) })
  const quality = qualityQ.data
  const base = quality?.compare_modes.find((m) => m.mode === 'base')
  const reranked = quality?.compare_modes.find((m) => m.mode === 'reranked')

  return <div>
    <div className='card'>
      <h2>Model Quality</h2>
      <div>Качество candidate generator и второго слоя</div>
      <div className='filters'>
        <input value={patternTag} onChange={(e) => setPatternTag(e.target.value)} placeholder='pattern_tag' />
        <select value={includeUncertain} onChange={(e) => setIncludeUncertain(e.target.value as 'ignore'|'false'|'separate')}>
          <option value='ignore'>ignore uncertain</option>
          <option value='false'>uncertain as false</option>
          <option value='separate'>separate</option>
        </select>
      </div>
    </div>

    <div className='card-grid' style={{ marginTop: 12 }}>
      <div className='card'><b>Layer 1 (Base)</b><div>candidate generation</div></div>
      <div className='card'><b>Layer 2 (Reranker)</b><div>prioritization of reviewed candidates</div></div>
      <div className='card'><b>Goal</b><div>improve precision@K without losing candidate coverage</div></div>
    </div>

    <div className='card-grid' style={{ marginTop: 12 }}>
      <div className='card'><b>reviewed rows</b><div className='kpi-value'>{quality?.reviewed_rows ?? 0}</div></div>
      <div className='card'><b>precision reviewed base/reranked</b><div className='kpi-value'>{base?.precision_reviewed ?? '—'} / {reranked?.precision_reviewed ?? '—'}</div></div>
      <div className='card'><b>lift vs base</b><div className='kpi-value'>{quality?.lift_vs_base ?? '—'}</div></div>
    </div>

    {!quality?.compare_modes?.length ? <div className='card' style={{ marginTop: 12 }}>No feedback data for quality metrics yet.</div> : <>
      <PrecisionAtKChart modes={quality.compare_modes} />
      <div className='card-grid'>
        <ReviewFunnel metrics={base} />
        <ScoreHistogram mode={base} />
        <ScoreBucketChart mode={base} />
      </div>
      <div className='card-grid'>
        <FalsePositivesByCategoryChart mode={base} />
        <FalsePositivesByClusterChart mode={base} />
      </div>
      <VersionComparisonTable versions={quality.versions} />
    </>}

    <UnflaggedAuditPanel />
  </div>
}
