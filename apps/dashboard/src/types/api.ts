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
