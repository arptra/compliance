from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def _normalized_centroids(cluster_centers: np.ndarray) -> np.ndarray:
    return normalize(cluster_centers)


def max_similarity_to_centroids(x: csr_matrix, centroids: np.ndarray) -> np.ndarray:
    sims = cosine_similarity(x, centroids)
    return sims.max(axis=1)


def novelty_threshold_from_baseline(max_sims: np.ndarray, percentile: float) -> float:
    return float(np.percentile(max_sims, percentile))


def compute_novelty_flags(x_december: csr_matrix, centroids: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    max_sims = max_similarity_to_centroids(x_december, centroids)
    return max_sims, (max_sims < threshold).astype(int)


def emerging_terms(
    x_baseline: csr_matrix,
    x_december: csr_matrix,
    feature_names: np.ndarray,
    top_n: int = 30,
) -> pd.DataFrame:
    baseline_mean = np.asarray(x_baseline.mean(axis=0)).ravel()
    december_mean = np.asarray(x_december.mean(axis=0)).ravel()
    lift = (december_mean + 1e-9) / (baseline_mean + 1e-9)
    support = december_mean * lift
    idx = np.argsort(support)[::-1][:top_n]
    return pd.DataFrame(
        {
            "term": feature_names[idx],
            "baseline_tfidf_mean": baseline_mean[idx],
            "december_tfidf_mean": december_mean[idx],
            "lift": lift[idx],
        }
    )


def cluster_novel_complaints(
    x_novel: csr_matrix,
    texts: List[str],
    feature_names: np.ndarray,
    cfg: Dict,
) -> pd.DataFrame:
    if x_novel.shape[0] == 0:
        return pd.DataFrame(columns=["cluster_id", "size", "top_terms", "example_messages"])

    k = min(cfg["novelty"].get("novel_december_subclusters", 8), max(1, x_novel.shape[0]))
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=cfg.get("random_state", 42),
        n_init="auto",
    )
    labels = model.fit_predict(x_novel)

    rows = []
    top_n = cfg["clustering"].get("top_terms", 12)
    ex_n = cfg["novelty"].get("examples_per_novel_cluster", 10)
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        centroid = model.cluster_centers_[cid]
        term_ids = np.argsort(centroid)[::-1][:top_n]
        rows.append(
            {
                "cluster_id": cid,
                "size": int(len(idx)),
                "top_terms": ", ".join(feature_names[term_ids]),
                "example_messages": " || ".join(texts[i][:200] for i in idx[:ex_n]),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def save_novelty_reports(
    emerging_df: pd.DataFrame,
    novel_clusters_df: pd.DataFrame,
    reports_dir: str | Path,
) -> None:
    rdir = Path(reports_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    emerging_df.to_csv(rdir / "emerging_terms.csv", index=False)
    novel_clusters_df.to_csv(rdir / "december_novel_complaints_clusters.csv", index=False)
