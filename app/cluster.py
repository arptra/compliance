from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocess import TextPreprocessor


def train_cluster_model(x_baseline: csr_matrix, cfg: Dict) -> MiniBatchKMeans:
    ccfg = cfg["clustering"]
    model = MiniBatchKMeans(
        n_clusters=ccfg.get("n_clusters", 30),
        batch_size=ccfg.get("batch_size", 4096),
        random_state=cfg.get("random_state", 42),
        n_init="auto",
    )
    model.fit(x_baseline)
    return model


def assign_clusters(x: csr_matrix, model: MiniBatchKMeans) -> Tuple[np.ndarray, np.ndarray]:
    cluster_ids = model.predict(x)
    sims = cosine_similarity(x, model.cluster_centers_)
    max_sims = sims.max(axis=1)
    return cluster_ids, max_sims


def _top_terms_ctfidf(
    texts: List[str],
    cluster_ids: np.ndarray,
    n_clusters: int,
    preprocessor: TextPreprocessor,
    top_n: int,
) -> dict[int, List[str]]:
    cluster_docs: List[str] = []
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        cluster_docs.append(" ".join(texts[i] for i in idx))

    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=preprocessor.analyzer,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=50_000,
        sublinear_tf=True,
        lowercase=False,
    )
    x_cluster = vec.fit_transform(cluster_docs)
    feat = vec.get_feature_names_out()

    top_terms: dict[int, List[str]] = {}
    for cid in range(n_clusters):
        row = x_cluster[cid].toarray().ravel()
        ranked = np.argsort(row)[::-1]
        terms: List[str] = []
        for rid in ranked:
            if row[rid] <= 0:
                continue
            term = feat[rid]
            if preprocessor.is_good_term(term):
                terms.append(term)
            if len(terms) >= top_n:
                break
        top_terms[cid] = terms
    return top_terms


def cluster_summaries(
    x_baseline: csr_matrix,
    texts: List[str],
    cluster_ids: np.ndarray,
    model: MiniBatchKMeans,
    feature_names: np.ndarray,
    cfg: Dict,
    preprocessor: TextPreprocessor,
) -> pd.DataFrame:
    ccfg = cfg["clustering"]
    scfg = cfg.get("summaries", {})
    top_n = scfg.get("top_terms_per_cluster", ccfg.get("top_terms", 12))
    ex_n = scfg.get("examples_per_cluster", ccfg.get("examples_per_cluster", 8))
    use_ctfidf = scfg.get("use_ctfidf", True)

    ctfidf_terms = (
        _top_terms_ctfidf(texts, cluster_ids, model.n_clusters, preprocessor, top_n)
        if use_ctfidf
        else {}
    )

    rows = []
    for cid in range(model.n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            rows.append({"cluster_id": cid, "size": 0, "top_terms": "", "example_messages": ""})
            continue

        terms = ctfidf_terms.get(cid, [])
        if not terms:
            centroid = model.cluster_centers_[cid]
            term_ids = np.argsort(centroid)[::-1]
            filtered = []
            for tid in term_ids:
                term = feature_names[tid]
                if preprocessor.is_good_term(term):
                    filtered.append(term)
                if len(filtered) >= top_n:
                    break
            terms = filtered

        examples = " || ".join(texts[i][:200] for i in idx[:ex_n])
        rows.append(
            {
                "cluster_id": cid,
                "size": int(len(idx)),
                "top_terms": ", ".join(terms),
                "example_messages": examples,
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def save_cluster_artifacts(
    model: MiniBatchKMeans,
    summary_df: pd.DataFrame,
    model_dir: str | Path,
    reports_dir: str | Path,
) -> None:
    model_path = Path(model_dir) / "cluster_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(reports_path / "cluster_summaries.csv", index=False)
