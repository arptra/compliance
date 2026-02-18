from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity


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


def cluster_summaries(
    x_baseline: csr_matrix,
    texts: List[str],
    cluster_ids: np.ndarray,
    model: MiniBatchKMeans,
    feature_names: np.ndarray,
    cfg: Dict,
) -> pd.DataFrame:
    ccfg = cfg["clustering"]
    top_n = ccfg.get("top_terms", 12)
    ex_n = ccfg.get("examples_per_cluster", 8)
    rows = []
    for cid in range(model.n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            rows.append({"cluster_id": cid, "size": 0, "top_terms": "", "example_messages": ""})
            continue
        centroid = model.cluster_centers_[cid]
        term_ids = np.argsort(centroid)[::-1][:top_n]
        top_terms = ", ".join(feature_names[term_ids])
        examples = " || ".join(texts[i][:200] for i in idx[:ex_n])
        rows.append(
            {
                "cluster_id": cid,
                "size": int(len(idx)),
                "top_terms": top_terms,
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
