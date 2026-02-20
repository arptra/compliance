from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor


def compute_novelty_scores(x_base, x_new, method: str, svd_components: int, kmeans_k: int, threshold_percentile: float):
    svd = TruncatedSVD(n_components=min(svd_components, x_base.shape[1] - 1))
    z_base = svd.fit_transform(x_base)
    z_new = svd.transform(x_new)

    if method == "lof":
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(z_base)
        base_scores = -lof.decision_function(z_base)
        new_scores = -lof.decision_function(z_new)
    else:
        km = KMeans(n_clusters=min(kmeans_k, len(z_base)), random_state=42, n_init="auto")
        km.fit(z_base)
        base_scores = np.min(km.transform(z_base), axis=1)
        new_scores = np.min(km.transform(z_new), axis=1)
    thr = np.percentile(base_scores, threshold_percentile)
    return new_scores, thr, z_new


def cluster_novel_texts(z_new: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return np.full(len(z_new), -1)
    z = z_new[idx]
    n_clusters = min(max(2, len(idx) // 10), 10)
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(z)
    out = np.full(len(z_new), -1)
    out[idx] = labels
    return out


def summarize_new_topics(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_novel"]].groupby("cluster_id").agg(count=("row_id", "count")).reset_index()
