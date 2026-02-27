from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocess import TextPreprocessor


def _compute_centers_from_labels(x: csr_matrix, labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)
    centers = []
    for cid in uniq:
        mask = labels == cid
        if mask.sum() == 0:
            continue
        centers.append(np.asarray(x[mask].mean(axis=0)).ravel())
    return np.vstack(centers) if centers else np.zeros((1, x.shape[1]))


def train_cluster_model(x_baseline: csr_matrix, cfg: Dict):
    ccfg = cfg["clustering"]
    model_name = ccfg.get("model", "minibatch_kmeans")
    rs = cfg.get("random_state", 42)

    if model_name == "kmeans":
        model = KMeans(
            n_clusters=ccfg.get("n_clusters", 30),
            random_state=rs,
            n_init="auto",
        )
    elif model_name == "birch":
        model = Birch(
            threshold=ccfg.get("birch_threshold", 0.5),
            branching_factor=ccfg.get("birch_branching_factor", 50),
            n_clusters=ccfg.get("n_clusters", 30),
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=ccfg.get("n_clusters", 30),
            batch_size=ccfg.get("batch_size", 4096),
            random_state=rs,
            n_init="auto",
        )

    model.fit(x_baseline)

    # Ensure we always have cluster_centers_ for downstream cosine-sim calculations
    if not hasattr(model, "cluster_centers_"):
        labels = model.predict(x_baseline)
        model.cluster_centers_ = _compute_centers_from_labels(x_baseline, labels)

    return model


def assign_clusters(x: csr_matrix, model) -> Tuple[np.ndarray, np.ndarray]:
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
    model,
    feature_names: np.ndarray,
    cfg: Dict,
    preprocessor: TextPreprocessor,
) -> pd.DataFrame:
    ccfg = cfg["clustering"]
    scfg = cfg.get("summaries", {})
    top_n = scfg.get("top_terms_per_cluster", ccfg.get("top_terms", 12))
    ex_n = scfg.get("examples_per_cluster", ccfg.get("examples_per_cluster", 8))
    use_ctfidf = scfg.get("use_ctfidf", True)

    n_clusters = len(np.unique(cluster_ids))
    ctfidf_terms = (
        _top_terms_ctfidf(texts, cluster_ids, n_clusters, preprocessor, top_n)
        if use_ctfidf
        else {}
    )

    rows = []
    for cid in sorted(np.unique(cluster_ids)):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            rows.append({"cluster_id": int(cid), "size": 0, "top_terms": "", "example_messages": ""})
            continue

        terms = ctfidf_terms.get(int(cid), [])
        if not terms:
            centroid = model.cluster_centers_[int(cid)] if int(cid) < model.cluster_centers_.shape[0] else model.cluster_centers_[0]
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
                "cluster_id": int(cid),
                "size": int(len(idx)),
                "top_terms": ", ".join(terms),
                "example_messages": examples,
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def save_cluster_artifacts(
    model,
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
