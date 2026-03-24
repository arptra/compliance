import { z } from 'zod'

export const kpiSchema = z.object({ key: z.string(), value: z.number(), delta: z.number().nullable().optional(), unit: z.string().optional() })
export const overviewSchema = z.object({
  kpis: z.array(kpiSchema),
  actual_vs_expected: z.array(z.object({ date: z.string(), actual: z.number(), expected: z.number().nullable().optional(), delta: z.number().nullable().optional() })),
  top_growth_categories: z.array(z.object({ category: z.string(), count: z.number(), share: z.number(), baseline_count: z.number().nullable().optional(), delta_abs: z.number().nullable().optional(), delta_pct: z.number().nullable().optional(), anomaly_score: z.number().nullable().optional(), pattern_score: z.number().nullable().optional() })),
  top_alert_categories: z.array(z.any()),
  executive_summary: z.string(),
  cards: z.array(z.any()),
  sparklines: z.record(z.array(z.number()))
})

export const executiveReportSchema = z.object({
  meta: z.object({ generated_at: z.string().optional(), date_from: z.string().nullable().optional(), date_to: z.string().nullable().optional(), compare_mode: z.string().optional(), baseline_date_from: z.string().nullable().optional(), baseline_date_to: z.string().nullable().optional(), categories: z.array(z.string()).optional(), include_examples: z.boolean().optional(), include_ownership: z.boolean().optional(), alert_rows: z.number().optional() }),
  kpis: z.object({
    total_complaints: z.number(),
    expected_complaints: z.number(),
    delta_abs: z.number(),
    delta_pct: z.number().nullable(),
    categories_above_baseline: z.number(),
    top_growth_category: z.string().nullable(),
    pattern_risk: z.object({ score: z.number().nullable(), label: z.enum(['low', 'medium', 'high', 'unavailable']), display_label: z.string(), status: z.enum(['success','warning','danger','neutral']), calc_mode: z.enum(['full','state_only','alerts_only','unavailable']) }),
    pattern_risk_score: z.number().nullable().optional(),
    pattern_risk_label: z.string().optional(),
    pattern_risk_display_label: z.string().optional(),
    pattern_risk_status: z.string().optional(),
    primary_area: z.object({ label: z.string(), confidence_note: z.string() }).nullable(),
  }),
  charts: z.object({
    actual_expected: z.array(z.object({ date: z.string(), actual: z.number(), expected: z.number(), delta: z.number(), delta_pct: z.number().nullable() })),
    category_contribution: z.array(z.object({ category: z.string(), actual: z.number(), expected: z.number(), delta: z.number(), contribution_pct: z.number() })),
    category_priority: z.array(z.object({ category: z.string(), actual: z.number(), expected: z.number(), delta: z.number(), delta_pct: z.number().nullable(), priority: z.enum(['high', 'medium', 'low']) })),
    alert_examples: z.array(z.object({ text: z.string(), category: z.string(), reason: z.string(), priority: z.enum(['high', 'medium', 'low']), score: z.number().nullable() })),
  }),
  summary: z.object({ headline: z.string(), bullets: z.array(z.string()), recommended_actions: z.array(z.string()) }),
  definitions: z.record(z.string()),
  export: z.object({ markdown: z.string(), html: z.string() }),
})

export type OverviewResponse = z.infer<typeof overviewSchema>
export type ExecutiveReportResponse = z.infer<typeof executiveReportSchema>
export type ExecutiveKpis = ExecutiveReportResponse['kpis']
export type ActualExpectedPoint = ExecutiveReportResponse['charts']['actual_expected'][number]
export type ContributionRow = ExecutiveReportResponse['charts']['category_contribution'][number]
export type CategoryPriorityRow = ExecutiveReportResponse['charts']['category_priority'][number]
export type AlertExampleCard = ExecutiveReportResponse['charts']['alert_examples'][number]
export type SummaryBlock = ExecutiveReportResponse['summary']

export type FeedbackDatasetItem = {
  id?: number | null
  row_id: string
  pattern_tag?: string | null
  verdict: 'true' | 'false' | 'uncertain'
  reviewer?: string | null
  review_date?: string | null
  reason_code?: string | null
  comment?: string | null
  base_score?: number | null
  rerank_score?: number | null
  category?: string | null
  subcategory?: string | null
  date_from?: string | null
  date_to?: string | null
  model_version?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type FeedbackDatasetSummary = {
  reviewed_rows: number
  true_count: number
  false_count: number
  uncertain_count: number
  precision_reviewed: number | null
  active_model_version: string | null
}

export type FeedbackDatasetResponse = {
  items: FeedbackDatasetItem[]
  total: number
  page: number
  page_size: number
  summary: FeedbackDatasetSummary
}

export type ReviewDatasetFiltersState = {
  pattern_tag?: string
  reviewer?: string
  verdict?: string
  category?: string
  subcategory?: string
  reason_code?: string
  model_version?: string
  date_from?: string
  date_to?: string
  q?: string
  page?: number
  page_size?: number
  sort_by?: string
  sort_order?: string
}

export type PrecisionAtKItem = { k: number; precision: number | null }
export type ScoreBucketItem = { bucket: string; reviewed_count: number; true_count: number; precision: number | null }
export type GroupPrecisionItem = { name: string; reviewed_count: number; true_count: number; false_count: number; precision: number | null }
export type ModelVersionMetricsItem = { version_id: string; train_rows: number | null; precision_reviewed: number | null; precision_at_50: number | null; precision_at_100: number | null; active: boolean }
export type ModeMetrics = {
  mode: 'base' | 'calibrated' | 'reranked'
  score_column: string
  reviewed_rows: number
  true_count: number
  false_count: number
  uncertain_count: number
  precision_reviewed: number | null
  precision_at: PrecisionAtKItem[]
  average_score_true: number | null
  average_score_false: number | null
  by_score_bucket: ScoreBucketItem[]
  by_category: GroupPrecisionItem[]
  by_cluster: GroupPrecisionItem[]
}

export type ModelQualityResponse = {
  reviewed_rows: number
  compare_modes: ModeMetrics[]
  lift_vs_base: number | null
  versions: ModelVersionMetricsItem[]
}

export type UnflaggedAuditEstimate = {
  reviewed_in_sample: number
  true_in_sample: number
  estimated_hidden_positive_rate: number | null
  estimated_hidden_positives_in_unflagged_pool: number | null
  note: string
}
export type UnflaggedAuditRow = { row_id: string; review_verdict?: 'true' | 'false' | 'uncertain' | null; reviewer?: string | null; reviewed_at?: string | null; comment?: string | null }
export type UnflaggedAuditSample = {
  sample_id: string
  created_at: string
  pattern_tag?: string | null
  date_from?: string | null
  date_to?: string | null
  sample_size: number
  source_pool_size: number
  query_meta_json?: string | null
  status: string
  rows: UnflaggedAuditRow[]
  estimate?: UnflaggedAuditEstimate | null
}
