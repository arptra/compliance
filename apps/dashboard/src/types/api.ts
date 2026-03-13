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
export type OverviewResponse = z.infer<typeof overviewSchema>
