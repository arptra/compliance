from pathlib import Path

import yaml

from complaints_trends.config import load_config


def test_llm_env_overrides(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "input": {
            "input_dir": "data/raw",
            "file_glob": "*.xlsx",
            "month_source": "filename",
            "month_regex": r"(\\d{4})-(\\d{2})",
            "month_column": None,
            "id_column": None,
            "signal_columns": ["dialog_text"],
            "dialog_column": "dialog_text",
            "encoding": "utf-8",
        },
        "client_first_extraction": {
            "enabled": True,
            "client_markers": ["CLIENT"],
            "operator_markers": ["OPERATOR"],
            "chatbot_markers": ["CHATBOT"],
            "stop_on_markers": ["OPERATOR"],
            "fallback_mode": "first_paragraph",
            "fallback_first_n_chars": 600,
            "min_client_len": 20,
            "take_second_client_if_too_short": True,
        },
        "pii": {
            "enabled": True,
            "replace_email": "<EMAIL>",
            "replace_phone": "<PHONE>",
            "replace_url": "<URL>",
            "replace_card": "<CARD>",
            "replace_account": "<ACCOUNT>",
        },
        "llm": {
            "enabled": True,
            "mode": "mtls",
            "base_url": "https://from-config",
            "ca_bundle_file": "config-ca",
            "cert_file": "config-cert",
            "key_file": "config-key",
            "key_file_password_env": "GIGACHAT_KEY_PASSWORD",
            "verify_ssl_certs": True,
            "model": "GigaChat",
            "max_workers": 1,
            "batch_size": 1,
            "max_text_chars": 100,
            "cache_db": "data/interim/cache.sqlite",
            "prompt_version": "v1",
        },
        "prepare": {"pilot_month": None, "pilot_limit": 100, "output_parquet": "a.parquet", "pilot_parquet": "b.parquet", "pilot_review_xlsx": "c.xlsx"},
        "training": {
            "text_field": "client_first_message",
            "complaint_threshold": 0.5,
            "vectorizer": {"word_ngram": [1, 2], "char_ngram": [3, 5], "max_features_word": 1000, "max_features_char": 1000, "min_df": 1, "max_df": 1.0},
            "classifier": {"complaint": "logreg", "category": "linearsvc"},
            "validation": {"split_mode": "time", "val_month": None},
            "model_dir": "models",
        },
        "analysis": {"novelty": {"enabled": True, "method": "kmeans_distance", "svd_components": 10, "kmeans_k": 3, "threshold_percentile": 95, "min_cluster_size": 2}, "reports_dir": "reports"},
        "files": {"deny_tokens_path": "configs/deny_tokens.txt", "extra_stopwords_path": "configs/extra_stopwords.txt", "categories_seed_path": "configs/categories_seed.yaml"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://from-env")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE_FILE", "env-ca")
    monkeypatch.setenv("GIGACHAT_CERT_FILE", "env-cert")
    monkeypatch.setenv("GIGACHAT_KEY_FILE", "env-key")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2")
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")

    loaded = load_config(cfg_path)
    assert loaded.llm.base_url == "https://from-env"
    assert loaded.llm.ca_bundle_file == "env-ca"
    assert loaded.llm.cert_file == "env-cert"
    assert loaded.llm.key_file == "env-key"
    assert loaded.llm.model == "GigaChat-2"
    assert loaded.llm.verify_ssl_certs is False
