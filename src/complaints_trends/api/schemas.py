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
    prepare_service_columns: list[str] = Field(default_factory=list)


class GigaChatTransportArtifact(BaseModel):
    label: str
    path: str
    exists: bool


class GigaChatTransportStatus(BaseModel):
    name: Literal["mtls", "token"]
    title: str
    description: str
    active: bool = False
    configured: bool = False
    ready: bool = False
    base_url: str
    oauth_url: str | None = None
    artifacts: list[GigaChatTransportArtifact] = Field(default_factory=list)
    message: str = ""


class GigaChatTransportStatusResponse(BaseModel):
    configured_mode: Literal["mtls", "tls", "token"]
    model: str
    transports: list[GigaChatTransportStatus] = Field(default_factory=list)


class GigaChatTransportProbeRequest(BaseModel):
    transport: Literal["mtls", "token"]


class GigaChatTransportProbeResponse(BaseModel):
    transport: Literal["mtls", "token"]
    ok: bool
    base_url: str
    oauth_url: str | None = None
    model: str
    message: str
    models: list[str] = Field(default_factory=list)


class GigaChatLabSettingOption(BaseModel):
    value: str
    label: str


class GigaChatLabSettingField(BaseModel):
    key: str
    label: str
    input_type: Literal["text", "textarea", "number", "boolean", "select"]
    section: str
    help_text: str | None = None
    value: Any = None
    options: list[GigaChatLabSettingOption] = Field(default_factory=list)


class GigaChatLabSettingsResponse(BaseModel):
    title: str = "GigaChat Lab Settings"
    fields: list[GigaChatLabSettingField] = Field(default_factory=list)
    saved_at: datetime | None = None


class GigaChatLabSettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class GigaChatWorkbookSheetPreview(BaseModel):
    name: str
    rows_total: int = 0
    column_count: int = 0
    columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)


class GigaChatWorkbookUploadResponse(BaseModel):
    upload_id: str
    filename: str
    file_format: Literal["excel", "csv"]
    sheet_count: int = 0
    sheets: list[GigaChatWorkbookSheetPreview] = Field(default_factory=list)


class GigaChatWorkbookSelectSheetRequest(BaseModel):
    sheet_name: str
    row_limit: int = 200


class GigaChatWorkbookSheetDataResponse(BaseModel):
    upload_id: str
    filename: str
    file_format: Literal["excel", "csv"]
    sheet_name: str
    total_rows: int = 0
    rendered_rows: int = 0
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class GigaChatFinalPromptRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)


class GigaChatFinalPromptResponse(BaseModel):
    generated_at: datetime
    source_columns: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    saved: bool = False
    saved_path: str | None = None


class GigaChatLabRowRunRequest(BaseModel):
    transport: Literal["mtls", "token"] = "mtls"
    values: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    row: dict[str, Any] = Field(default_factory=dict)
    payload_override: dict[str, Any] | None = None
    count_tokens: bool = False


class GigaChatLabRowRunResponse(BaseModel):
    transport: Literal["mtls", "token"]
    request_payload: dict[str, Any] = Field(default_factory=dict)
    response_raw: str = ""
    response_json: dict[str, Any] | list[Any] | None = None
    parse_ok: bool = False
    request_token_count: int | None = None


class GigaChatAnnotatedExportRow(BaseModel):
    row_index: int | None = None
    classification: str = ""
    tags: list[str] = Field(default_factory=list)
    source_row: dict[str, Any] = Field(default_factory=dict)


class GigaChatAnnotatedExportRequest(BaseModel):
    filename: str = "annotated.xlsx"
    sheet_name: str = "Разметка"
    source_columns: list[str] = Field(default_factory=list)
    rows: list[GigaChatAnnotatedExportRow] = Field(default_factory=list)


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
    allowed: bool = True
    reason: str | None = None
    upload_id: str | None = None


class AlertRowResponse(BaseModel):
    rows: list[dict[str, Any]]
    scoring_mode_requested: Literal["base", "calibrated", "reranked"] = "base"
    scoring_mode_effective: Literal["base", "calibrated", "reranked"] = "base"
    reranker_available: bool = False
    active_calibrator_version: str | None = None


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


class PreparationJobSummary(BaseModel):
    upload_id: str
    original_filename: str
    stored_path: str
    uploaded_at: datetime
    status: Literal["uploaded", "queued", "running", "succeeded", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    rows_total: int = 0
    prepared_rows: int = 0
    complaints_rows: int = 0
    date_min: str | None = None
    date_max: str | None = None
    output_prepared_parquet: str | None = None
    merged_into_main: bool = False
    available_for_pattern_monitor: bool = False
    pattern_monitor_tag: str | None = None


class PreparationUploadResponse(BaseModel):
    upload_id: str
    filename: str
    uploaded_at: datetime
    status: str


class PreparationRunResponse(BaseModel):
    upload_id: str
    status: str
    error_message: str | None = None


class PreparationJobsResponse(BaseModel):
    jobs: list[PreparationJobSummary]


class PreparationPreviewResponse(BaseModel):
    upload_id: str
    filename: str
    status: str
    rows_total: int = 0
    date_min: str | None = None
    date_max: str | None = None
    available_columns: list[str] = Field(default_factory=list)


class PatternMonitorPresetPayload(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    month: str | None = None
    upload_id: str
    pattern_tag: str | None = None
    label_source: str = "llm"
    source_filename: str | None = None


class PatternMonitorPresetResponse(BaseModel):
    allowed: bool
    reason: str | None = None
    pattern_monitor_preset: PatternMonitorPresetPayload | None = None


class RunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    status: Literal["success", "error"]
    started_at: datetime
    finished_at: datetime
    outputs: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None


class FeedbackCreate(BaseModel):
    row_id: str
    pattern_tag: str | None = None
    verdict: Literal["true", "false", "uncertain"]
    reviewer: str | None = None
    review_date: str | None = None
    reason_code: str | None = None
    comment: str | None = None
    base_score: float | None = None
    rerank_score: float | None = None
    category: str | None = None
    category_label_ru: str | None = None
    subcategory: str | None = None
    subcategory_label_ru: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    model_version: str | None = None


class FeedbackBulkCreate(BaseModel):
    rows: list[FeedbackCreate]


class FeedbackItem(FeedbackCreate):
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class FeedbackSummaryResponse(BaseModel):
    reviewed_rows: int = 0
    true_count: int = 0
    false_count: int = 0
    uncertain_count: int = 0
    precision_reviewed: float | None = None
    precision_at_50: float | None = None
    precision_at_100: float | None = None
    by_category: list[dict[str, Any]] = Field(default_factory=list)
    by_cluster: list[dict[str, Any]] = Field(default_factory=list)
    by_score_bucket: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackDatasetItem(FeedbackItem):
    pass


class FeedbackDatasetSummary(BaseModel):
    reviewed_rows: int = 0
    true_count: int = 0
    false_count: int = 0
    uncertain_count: int = 0
    precision_reviewed: float | None = None
    active_model_version: str | None = None


class FeedbackDatasetResponse(BaseModel):
    items: list[FeedbackDatasetItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    summary: FeedbackDatasetSummary = Field(default_factory=FeedbackDatasetSummary)


class PrecisionAtKItem(BaseModel):
    k: int
    precision: float | None = None


class ScoreBucketItem(BaseModel):
    bucket: str
    reviewed_count: int
    true_count: int
    precision: float | None = None


class CategoryPrecisionItem(BaseModel):
    name: str
    label_ru: str | None = None
    reviewed_count: int
    true_count: int
    false_count: int
    precision: float | None = None


class ClusterPrecisionItem(CategoryPrecisionItem):
    pass


class ModelVersionMetricsItem(BaseModel):
    version_id: str
    train_rows: int | None = None
    precision_reviewed: float | None = None
    precision_at_50: float | None = None
    precision_at_100: float | None = None
    active: bool = False


class ModeMetrics(BaseModel):
    mode: Literal["base", "calibrated", "reranked"]
    score_column: str
    reviewed_rows: int = 0
    true_count: int = 0
    false_count: int = 0
    uncertain_count: int = 0
    precision_reviewed: float | None = None
    precision_at: list[PrecisionAtKItem] = Field(default_factory=list)
    average_score_true: float | None = None
    average_score_false: float | None = None
    by_score_bucket: list[ScoreBucketItem] = Field(default_factory=list)
    by_category: list[CategoryPrecisionItem] = Field(default_factory=list)
    by_cluster: list[ClusterPrecisionItem] = Field(default_factory=list)


class ModelQualityResponse(BaseModel):
    reviewed_rows: int = 0
    compare_modes: list[ModeMetrics] = Field(default_factory=list)
    lift_vs_base: float | None = None
    versions: list[ModelVersionMetricsItem] = Field(default_factory=list)


class UnflaggedAuditRow(BaseModel):
    row_id: str
    review_verdict: Literal["true", "false", "uncertain"] | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    comment: str | None = None


class UnflaggedAuditEstimateResponse(BaseModel):
    reviewed_in_sample: int = 0
    true_in_sample: int = 0
    estimated_hidden_positive_rate: float | None = None
    estimated_hidden_positives_in_unflagged_pool: float | None = None
    note: str = ""


class UnflaggedAuditSample(BaseModel):
    sample_id: str
    created_at: str
    pattern_tag: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sample_size: int
    source_pool_size: int
    query_meta_json: str | None = None
    status: str
    rows: list[UnflaggedAuditRow] = Field(default_factory=list)
    estimate: UnflaggedAuditEstimateResponse | None = None


class UnflaggedAuditCreateRequest(BaseModel):
    pattern_tag: str = "latest"
    date_from: str | None = None
    date_to: str | None = None
    sample_size: int = 100
    random_seed: int | None = None


class UnflaggedAuditReviewRequest(BaseModel):
    rows: list[UnflaggedAuditRow]


class CalibratorTrainRequest(BaseModel):
    pattern_tag: str = "latest"
    date_from: str | None = None
    date_to: str | None = None
    reviewer: str | None = None
    algorithm: str | None = "logistic_regression"
    activate_if_better: bool = True


class CalibratorVersionResponse(BaseModel):
    version_id: str
    created_at: str
    status: str
    algorithm: str
    metrics_json: dict[str, Any] | None = None
    artifact_path: str | None = None
    train_rows: int | None = None
    active: int = 0
    notes: str | None = None
