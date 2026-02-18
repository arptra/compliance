from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(cfg: Dict) -> TfidfVectorizer:
    vcfg = cfg["vectorizer"]
    ngram_range = tuple(vcfg.get("ngram_range", [1, 2]))
    analyzer = "char_wb" if cfg["preprocess"].get("use_char_ngrams", False) else "word"
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=vcfg.get("min_df", 3),
        max_df=vcfg.get("max_df", 0.5),
        max_features=vcfg.get("max_features", 200_000),
        sublinear_tf=True,
        analyzer=analyzer,
    )


def fit_transform(
    baseline_texts: list[str],
    december_texts: list[str],
    cfg: Dict,
) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix]:
    vec = build_vectorizer(cfg)
    x_baseline = vec.fit_transform(baseline_texts)
    x_december = vec.transform(december_texts)
    return vec, x_baseline, x_december


def save_vectorizer(vectorizer: TfidfVectorizer, model_dir: str | Path) -> Path:
    model_path = Path(model_dir) / "vectorizer.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_path)
    return model_path
