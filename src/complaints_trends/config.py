from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AppBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class InputConfig(AppBaseModel):
    input_dir: str
    file_glob: str = "*.xlsx"
    file_names: list[str] | None = None
    datetime_column: str = "created_at"
    datetime_format: str | None = None
    id_column: str | None = None
    signal_columns: list[str]
    dialog_column: str | None = "dialog_text"
    dialog_columns: list[str] | None = None
    encoding: str = "utf-8"


class ClientFirstConfig(AppBaseModel):
    enabled: bool = True
    client_markers: list[str]
    operator_markers: list[str]
    chatbot_markers: list[str]
    stop_on_markers: list[str]
    fallback_mode: Literal["first_paragraph", "first_n_chars"] = "first_paragraph"
    fallback_first_n_chars: int = 600
    min_client_len: int = 20
    take_second_client_if_too_short: bool = True


class PIIConfig(AppBaseModel):
    enabled: bool = True
    replace_email: str = "<EMAIL>"
    replace_phone: str = "<PHONE>"
    replace_url: str = "<URL>"
    replace_card: str = "<CARD>"
    replace_account: str = "<ACCOUNT>"


class LLMConfig(AppBaseModel):
    enabled: bool = True
    mode: Literal["mtls", "tls"] = "mtls"
    base_url: str
    ca_bundle_file: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    key_file_password_env: str | None = None
    verify_ssl_certs: bool = True
    model: str = "GigaChat"
    max_workers: int = 8
    batch_size: int = 20
    max_text_chars: int = 1200
    cache_db: str = "data/interim/gigachat_cache.sqlite"
    prompt_version: str = "v1"
    token_batch_size: int = 12000
    batch_mode: bool = False
    request_metrics_enabled: bool = True
    async_mode: bool = False
    parallel_mode: bool = False
    category_mode: Literal["taxonomy", "discover", "questions"] = "taxonomy"
    discovered_taxonomy_file: str = "data/interim/discovered_categories.json"
    questions_file: str = "configs/questions_categories.json"


class PrepareConfig(AppBaseModel):
    pilot_limit: int = 5000
    date_from: str | None = None
    date_to: str | None = None
    output_parquet: str
    pilot_parquet: str
    pilot_review_xlsx: str
    llm_payload_review_xlsx: str = "exports/llm_payload_review.xlsx"


class VectorizerConfig(AppBaseModel):
    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (3, 5)
    max_features_word: int = 200000
    max_features_char: int = 120000
    min_df: int = 5
    max_df: float = 0.6


class ClassifierConfig(AppBaseModel):
    complaint: Literal["logreg", "linearsvc"] = "logreg"
    category: Literal["linearsvc", "logreg"] = "linearsvc"


class ValidationConfig(AppBaseModel):
    split_mode: Literal["time", "random"] = "time"
    val_from: str | None = None
    val_to: str | None = None


class TrainingConfig(AppBaseModel):
    text_field: str = "client_first_message"
    complaint_threshold: float = 0.5
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model_dir: str = "models"


class NoveltyConfig(AppBaseModel):
    enabled: bool = True
    method: Literal["kmeans_distance", "lof"] = "kmeans_distance"
    svd_components: int = 200
    kmeans_k: int = 40
    threshold_percentile: float = 98
    min_cluster_size: int = 20


class NoveltyHuntConfig(AppBaseModel):
    enabled: bool = True

    candidate_pool: Literal["complaints_only", "score_band", "all"] = "score_band"
    complaint_score_min: float = 0.35
    complaint_score_max: float = 1.0
    include_other_category: bool = True
    include_low_margin: bool = True

    novelty_scope: Literal["global", "per_category", "both"] = "both"
    min_baseline_per_category: int = 200
    label_source: Literal["pred", "llm"] = "pred"

    text_field: str = "client_first_message"
    use_first_message_only: bool = True
    strip_system_speakers: bool = True

    vectorizer_source: Literal["trained", "fit_baseline"] = "trained"

    method_primary: Literal["knn_cosine", "lof", "kmeans_distance"] = "knn_cosine"
    method_secondary: Literal["none", "knn_cosine", "lof", "kmeans_distance"] = "kmeans_distance"
    combine: Literal["max", "mean", "weighted"] = "max"
    weight_primary: float = 0.7
    weight_secondary: float = 0.3
    svd_components: int = 200
    knn_k: int = 15
    kmeans_k: int = 40

    select_mode: Literal["percentile_base", "top_k", "threshold"] = "percentile_base"
    threshold_percentile: float = 98.0
    threshold_value: float = 0.5
    top_k: int = 300

    clustering: Literal["optics", "dbscan", "agglomerative"] = "optics"
    min_cluster_size: int = 15
    max_clusters: int = 12

    reports_dir: str = "reports"
    exports_dir: str = "exports"
    interim_dir: str = "data/interim"
    examples_per_cluster: int = 12

    llm_summary_enabled: bool = False
    llm_max_clusters: int = 8
    llm_examples_per_cluster: int = 10




class PatternMonitoringConfig(AppBaseModel):
    enabled: bool = True
    label_source: Literal["llm", "pred"] = "llm"
    normal_period: str | None = None
    event_period: str | None = None
    freq: Literal["D"] = "D"
    text_field: str = "client_first_message"
    use_first_message_only: bool = True
    strip_system_speakers: bool = True
    complaints_only: bool = True
    include_other_category: bool = False
    min_rows_per_category: int = 80
    min_normal_rows_per_category: int = 120
    min_event_rows_per_category: int = 60
    baseline_weekday_shrink_k: float = 5.0
    baseline_min_std: float = 1.0
    anomaly_z_threshold: float = 2.5
    anomaly_min_excess_total: int = 20
    anomaly_min_event_days: int = 3
    top_growth_categories: int = 12
    vectorizer_source: Literal["trained", "fit_normal"] = "trained"
    svd_components: int = 200
    random_state: int = 42
    within_category_method: Literal["knn_cosine", "centroid_delta"] = "knn_cosine"
    knn_k: int = 15
    per_day_seed_quota_mode: Literal["residual", "percent", "fixed"] = "residual"
    per_day_seed_percent: float = 0.3
    per_day_seed_fixed: int = 10
    min_total_seeds_per_category: int = 15
    max_total_seeds_per_category: int = 500
    cluster_method: Literal["optics", "dbscan", "agglomerative"] = "optics"
    min_cluster_size: int = 10
    max_clusters_per_category: int = 5
    score_w_event_similarity: float = 0.45
    score_w_normal_distance: float = 0.25
    score_w_category_anomaly: float = 0.30
    score_topk_event_neighbors: int = 10
    score_topk_normal_neighbors: int = 20
    daily_top_k_rows: int = 10
    daily_pressure_mode: Literal["sum_topk", "one_minus_prod"] = "sum_topk"
    state_alpha: float = 0.35
    row_alert_threshold: float = 0.65
    day_alert_threshold: float = 0.50
    interim_dir: str = "data/interim"
    exports_dir: str = "exports"
    reports_dir: str = "reports"


class AnalysisConfig(AppBaseModel):
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    novelty_hunt: NoveltyHuntConfig = Field(default_factory=NoveltyHuntConfig)
    pattern_monitoring: PatternMonitoringConfig = Field(default_factory=PatternMonitoringConfig)
    reports_dir: str = "reports"


class FilesConfig(AppBaseModel):
    deny_tokens_path: str
    extra_stopwords_path: str
    categories_seed_path: str


class ProjectConfig(AppBaseModel):
    input: InputConfig
    client_first_extraction: ClientFirstConfig
    pii: PIIConfig
    llm: LLMConfig
    prepare: PrepareConfig
    training: TrainingConfig
    analysis: AnalysisConfig
    files: FilesConfig


def _env_bool(name: str) -> bool | None:
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_llm_env_overrides(data: dict) -> dict:
    llm = data.setdefault("llm", {})
    env_map = {
        "GIGACHAT_BASE_URL": "base_url",
        "GIGACHAT_CA_BUNDLE_FILE": "ca_bundle_file",
        "GIGACHAT_CERT_FILE": "cert_file",
        "GIGACHAT_KEY_FILE": "key_file",
        "GIGACHAT_MODEL": "model",
    }
    for env_name, cfg_key in env_map.items():
        env_val = os.getenv(env_name)
        if env_val:
            llm[cfg_key] = env_val

    env_pwd_var = os.getenv("GIGACHAT_KEY_PASSWORD_ENV")
    if env_pwd_var:
        llm["key_file_password_env"] = env_pwd_var

    env_verify = _env_bool("GIGACHAT_VERIFY_SSL_CERTS")
    if env_verify is not None:
        llm["verify_ssl_certs"] = env_verify

    return data


def load_config(path: str | Path) -> ProjectConfig:
    load_dotenv(override=False)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _apply_llm_env_overrides(data)
    return ProjectConfig.model_validate(data)
