from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str = "complaints-trends-api"
    timestamp: datetime


class DatasetMeta(BaseModel):
    name: str
    path: str
    exists: bool
    rows: int | None = None


class AvailableTagsResponse(BaseModel):
    viz_tags: list[str]
    pattern_fit_tags: list[str]
    pattern_monitor_tags: list[str]


class MetaConfigResponse(BaseModel):
    prepare_output_parquet: str
    reports_dir: str
    interim_dir: str


class MetaDatasetsResponse(BaseModel):
    datasets: list[DatasetMeta]
    min_date: date | None = None
    max_date: date | None = None
    label_sources: list[str] = Field(default_factory=list)


class KPIItem(BaseModel):
    key: str
    value: float
    delta: float | None = None
    unit: str = "count"


class TimeSeriesPoint(BaseModel):
    date: date
    actual: float
    expected: float | None = None
    delta: float | None = None


class CategoryMetric(BaseModel):
    category: str
    count: float
    share: float
    baseline_count: float | None = None
    delta_abs: float | None = None
    delta_pct: float | None = None
    anomaly_score: float | None = None
    pattern_score: float | None = None


class OverviewResponse(BaseModel):
    kpis: list[KPIItem]
    actual_vs_expected: list[TimeSeriesPoint]
    top_growth_categories: list[CategoryMetric]
    top_alert_categories: list[CategoryMetric]
    executive_summary: str
    cards: list[dict[str, Any]]
    sparklines: dict[str, list[float]]


class CategoryTableResponse(BaseModel):
    rows: list[CategoryMetric]


class CategoryDetailResponse(BaseModel):
    category: str
    timeseries: list[TimeSeriesPoint]


class SubcategoryResponse(BaseModel):
    category: str
    subcategories: list[dict[str, Any]]


class ExamplesResponse(BaseModel):
    category: str | None = None
    examples: list[dict[str, Any]]


class HeatmapCell(BaseModel):
    dow: int
    hour: int
    value: float


class CompareResponse(BaseModel):
    actual_total: float
    baseline_total: float
    delta_abs: float
    delta_pct: float | None = None
    contributions: list[CategoryMetric]


class PatternFitSummaryResponse(BaseModel):
    tag: str
    growth_summary: list[dict[str, Any]]
    categories: list[str]


class ClusterProfileResponse(BaseModel):
    category: str
    clusters: list[dict[str, Any]]


class PatternMonitorSummaryPayload(BaseModel):
    scored_rows: int = 0
    alert_rows: int = 0
    pressure_days: int = 0
    latest_overall_pressure: float | None = None
    latest_smoothed_state: float | None = None
    pattern_risk_score: float | None = None
    pattern_risk_label: Literal["low", "medium", "high", "unavailable"] = "unavailable"
    pattern_risk_display_label: str = "Недоступно"
    pattern_risk_status: Literal["success", "warning", "danger", "neutral"] = "neutral"
    pattern_risk_calc_mode: Literal["full", "state_only", "alerts_only", "unavailable"] = "unavailable"


class PatternMonitorSummaryResponse(BaseModel):
    tag: str
    summary: PatternMonitorSummaryPayload


class AlertRowResponse(BaseModel):
    rows: list[dict[str, Any]]


class DailyPressureResponse(BaseModel):
    rows: list[dict[str, Any]]


class OverallStateResponse(BaseModel):
    rows: list[dict[str, Any]]


class ReportRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    output_format: Literal["json", "markdown", "html"] = "json"


class ReportResponse(BaseModel):
    report_type: str
    sections: list[dict[str, Any]]
    text_summary: str
    metrics: dict[str, Any]
    examples: list[dict[str, Any]] = Field(default_factory=list)
    markdown: str | None = None
    html: str | None = None


class ExecutiveReportRequest(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    compare_mode: Literal["previous_period", "same_weekday", "seasonal", "custom_range"] = "previous_period"
    baseline_date_from: str | None = None
    baseline_date_to: str | None = None
    categories: list[str] | None = None
    include_examples: bool = True
    include_ownership: bool = True
    pattern_tag: str = "latest"


class PatternRisk(BaseModel):
    score: float | None = None
    label: Literal["low", "medium", "high", "unavailable"] = "unavailable"
    display_label: str = "Недоступно"
    status: Literal["success", "warning", "danger", "neutral"] = "neutral"
    calc_mode: Literal["full", "state_only", "alerts_only", "unavailable"] = "unavailable"


class PrimaryArea(BaseModel):
    label: str
    confidence_note: str


class ExecutiveKpis(BaseModel):
    total_complaints: int
    expected_complaints: int
    delta_abs: int
    delta_pct: float | None = None
    categories_above_baseline: int
    top_growth_category: str | None = None
    pattern_risk: PatternRisk
    pattern_risk_score: float | None = None
    pattern_risk_label: str = "unavailable"
    pattern_risk_display_label: str = "Недоступно"
    pattern_risk_status: str = "neutral"
    primary_area: PrimaryArea | None = None


class ActualExpectedPoint(BaseModel):
    date: str
    actual: float
    expected: float
    delta: float
    delta_pct: float | None = None


class ContributionRow(BaseModel):
    category: str
    actual: float
    expected: float
    delta: float
    contribution_pct: float


class CategoryPriorityRow(BaseModel):
    category: str
    actual: float
    expected: float
    delta: float
    delta_pct: float | None = None
    priority: Literal["high", "medium", "low"]


class AlertExampleCard(BaseModel):
    text: str
    category: str
    reason: str
    priority: Literal["high", "medium", "low"]
    score: float | None = None


class SummaryBlock(BaseModel):
    headline: str
    bullets: list[str]
    recommended_actions: list[str]


class ExecutiveCharts(BaseModel):
    actual_expected: list[ActualExpectedPoint]
    category_contribution: list[ContributionRow]
    category_priority: list[CategoryPriorityRow]
    alert_examples: list[AlertExampleCard]


class ExecutiveExport(BaseModel):
    markdown: str
    html: str


class ExecutiveReportResponse(BaseModel):
    meta: dict[str, Any]
    kpis: ExecutiveKpis
    charts: ExecutiveCharts
    summary: SummaryBlock
    definitions: dict[str, str]
    export: ExecutiveExport


class RunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    status: Literal["success", "error"]
    started_at: datetime
    finished_at: datetime
    outputs: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
