from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
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


class ClientFirstConfig(BaseModel):
    enabled: bool = True
    client_markers: list[str]
    operator_markers: list[str]
    chatbot_markers: list[str]
    stop_on_markers: list[str]
    fallback_mode: Literal["first_paragraph", "first_n_chars"] = "first_paragraph"
    fallback_first_n_chars: int = 600
    min_client_len: int = 20
    take_second_client_if_too_short: bool = True


class PIIConfig(BaseModel):
    enabled: bool = True
    replace_email: str = "<EMAIL>"
    replace_phone: str = "<PHONE>"
    replace_url: str = "<URL>"
    replace_card: str = "<CARD>"
    replace_account: str = "<ACCOUNT>"


class LLMConfig(BaseModel):
    enabled: bool = True
    mode: str = "mtls"
    base_url: str
    ca_bundle_file: str
    cert_file: str
    key_file: str
    key_file_password_env: str | None = None
    verify_ssl_certs: bool = True
    model: str = "GigaChat"
    max_workers: int = 8
    batch_size: int = 20
    max_text_chars: int = 1200
    cache_db: str = "data/interim/gigachat_cache.sqlite"
    prompt_version: str = "v1"


class PrepareConfig(BaseModel):
    pilot_limit: int = 5000
    date_from: str | None = None
    date_to: str | None = None
    output_parquet: str
    pilot_parquet: str
    pilot_review_xlsx: str


class VectorizerConfig(BaseModel):
    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (3, 5)
    max_features_word: int = 200000
    max_features_char: int = 120000
    min_df: int = 5
    max_df: float = 0.6


class ClassifierConfig(BaseModel):
    complaint: Literal["logreg", "linearsvc"] = "logreg"
    category: Literal["linearsvc", "logreg"] = "linearsvc"


class ValidationConfig(BaseModel):
    split_mode: Literal["time", "random"] = "time"
    val_from: str | None = None
    val_to: str | None = None


class TrainingConfig(BaseModel):
    text_field: str = "client_first_message"
    complaint_threshold: float = 0.5
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model_dir: str = "models"


class NoveltyConfig(BaseModel):
    enabled: bool = True
    method: Literal["kmeans_distance", "lof"] = "kmeans_distance"
    svd_components: int = 200
    kmeans_k: int = 40
    threshold_percentile: float = 98
    min_cluster_size: int = 20


class AnalysisConfig(BaseModel):
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    reports_dir: str = "reports"


class FilesConfig(BaseModel):
    deny_tokens_path: str
    extra_stopwords_path: str
    categories_seed_path: str


class ProjectConfig(BaseModel):
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
