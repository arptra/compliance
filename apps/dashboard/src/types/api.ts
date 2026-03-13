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

export const overallTimeseriesSchema = z.object({
  actual: z.array(z.object({ date: z.string(), value: z.number() })),
  expected: z.array(z.object({ date: z.string(), value: z.number() })),
  delta: z.array(z.object({ date: z.string(), actual: z.number(), expected: z.number(), delta_abs: z.number(), delta_pct: z.number().nullable() })),
  cumulative: z.array(z.object({ date: z.string(), actual: z.number(), expected: z.number() })),
  summary: z.record(z.any())
})

export const categoryTimeseriesSchema = z.object({
  rows: z.array(z.object({ date: z.string(), category: z.string(), count: z.number(), share: z.number() })),
  resolved_categories: z.array(z.string()).optional(),
  used_other: z.boolean().optional()
})

export const heatmapSchema = z.object({
  weekday_hour: z.array(z.object({ dow: z.number(), hour: z.number(), value: z.number() })),
  calendar: z.array(z.object({ date: z.string(), value: z.number() }))
})

export const compareSummarySchema = z.object({
  summary: z.object({ actual_total: z.number(), baseline_total: z.number(), delta_abs: z.number(), delta_pct: z.number().nullable() }),
  contributions: z.array(z.object({ category: z.string(), actual_count: z.number(), expected_count: z.number(), delta_abs: z.number(), delta_pct: z.number().nullable(), share: z.number(), contribution_to_growth: z.number(), anomaly_score: z.number() }))
})

export type OverviewResponse = z.infer<typeof overviewSchema>
