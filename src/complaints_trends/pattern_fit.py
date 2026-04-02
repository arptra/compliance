from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from .config import ProjectConfig
from .pattern_common import (
    build_category_baseline,
    build_category_vector_space,
    build_daily_category_counts,
    build_text_clean,
    compute_growth_summary,
    get_label_columns,
    render_pattern_fit_report,
    select_event_seed_pool,
    select_period_rows,
    cluster_seed_pool,
)
from .pattern_paths import PatternFitPaths

logger = logging.getLogger(__name__)


def _pick_effective_label_source(df: pd.DataFrame, requested_source: str, complaints_only: bool, include_other_category: bool) -> str:
    def _signal(source: str) -> tuple[int, int]:
        is_complaint, category = get_label_columns(df, source)
        complaint_rows = int(is_complaint.sum())
        non_other_rows = int((category.fillna("OTHER").astype(str) != "OTHER").sum())
        return complaint_rows, non_other_rows

    requested_complaints, requested_non_other = _signal(requested_source)
    if not complaints_only and include_other_category:
        return requested_source

    has_requested_signal = True
    if complaints_only and requested_complaints == 0:
        has_requested_signal = False
    if not include_other_category and requested_non_other == 0:
        has_requested_signal = False
    if has_requested_signal:
        return requested_source

    fallback_source = "pred" if requested_source == "llm" else "llm"
    fallback_complaints, fallback_non_other = _signal(fallback_source)
    has_fallback_signal = True
    if complaints_only and fallback_complaints == 0:
        has_fallback_signal = False
    if not include_other_category and fallback_non_other == 0:
        has_fallback_signal = False

    if has_fallback_signal:
        logger.warning(
            "pattern-fit: requested label_source=%s has no usable rows (complaints=%s, non_other=%s); fallback to %s (complaints=%s, non_other=%s)",
            requested_source,
            requested_complaints,
            requested_non_other,
            fallback_source,
            fallback_complaints,
            fallback_non_other,
        )
        return fallback_source
    return requested_source


def run_pattern_fit(cfg: ProjectConfig, tag: str, normal_period: str, event_period: str, label_source: str) -> tuple[Path, Path, Path]:
    pm = cfg.analysis.pattern_monitoring
    paths = PatternFitPaths(tag=tag, interim_dir=pm.interim_dir, exports_dir=pm.exports_dir, reports_dir=pm.reports_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.export.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_parquet(cfg.prepare.output_parquet)
    effective_label_source = _pick_effective_label_source(base_df, label_source, pm.complaints_only, pm.include_other_category)
    is_complaint, category = get_label_columns(base_df, effective_label_source)
    base_df = base_df.copy()
    base_df["is_complaint"] = is_complaint
    base_df["category"] = category
    base_df["event_time"] = pd.to_datetime(base_df.get("event_time"), errors="coerce")
    base_df["month"] = base_df.get("month", base_df["event_time"].dt.strftime("%Y-%m"))

    normal = select_period_rows(base_df, normal_period)
    event = select_period_rows(base_df, event_period)
    if pm.complaints_only:
        normal = normal[normal["is_complaint"] == True]
        event = event[event["is_complaint"] == True]
    if not pm.include_other_category:
        normal = normal[normal["category"] != "OTHER"]
        event = event[event["category"] != "OTHER"]

    normal["text_original"] = normal.get(pm.text_field, normal.get("client_first_message", "")).fillna("").astype(str)
    event["text_original"] = event.get(pm.text_field, event.get("client_first_message", "")).fillna("").astype(str)
    normal["text_clean"] = build_text_clean(normal, cfg, pm.text_field, pm.use_first_message_only, pm.strip_system_speakers)
    event["text_clean"] = build_text_clean(event, cfg, pm.text_field, pm.use_first_message_only, pm.strip_system_speakers)

    normal.to_parquet(paths.root / "normal_rows.parquet", index=False)
    event.to_parquet(paths.root / "event_rows.parquet", index=False)

    counts_normal = build_daily_category_counts(normal)
    counts_event = build_daily_category_counts(event)
    daily_counts = pd.concat([counts_normal.assign(source_period="normal"), counts_event.assign(source_period="event")], ignore_index=True)
    daily_counts.to_parquet(paths.root / "daily_category_counts.parquet", index=False)

    baseline_event, weekday_stats = build_category_baseline(counts_normal, counts_event, pm.baseline_weekday_shrink_k, pm.baseline_min_std)
    baseline_event.to_parquet(paths.root / "category_baseline.parquet", index=False)

    growth = compute_growth_summary(
        baseline_event,
        anomaly_z_threshold=pm.anomaly_z_threshold,
        anomaly_min_excess_total=pm.anomaly_min_excess_total,
        anomaly_min_event_days=pm.anomaly_min_event_days,
        top_growth_categories=pm.top_growth_categories,
    )
    growth.to_parquet(paths.growth_summary, index=False)

    seed_rows = []
    cluster_members = []
    cluster_profiles: list[dict[str, Any]] = []
    fit_categories: dict[str, Any] = {}

    candidates = growth[growth["is_candidate"] == True]["category"].tolist()
    for cat in candidates:
        normal_c = normal[normal["category"] == cat].copy()
        event_c = event[event["category"] == cat].copy()
        if len(normal_c) < pm.min_normal_rows_per_category or len(event_c) < pm.min_event_rows_per_category:
            continue
        vs = build_category_vector_space(normal_c["text_clean"], event_c["text_clean"], cfg, pm.vectorizer_source, pm.svd_components, pm.random_state)
        z_normal, z_event = vs["z_normal"], vs["z_event"]

        if pm.within_category_method == "centroid_delta":
            centroid = np.mean(z_normal, axis=0, keepdims=True)
            novelty = cosine_distances(z_event, centroid).reshape(-1)
        else:
            k = max(1, min(pm.knn_k, len(z_normal)))
            nn = NearestNeighbors(metric="cosine", n_neighbors=k)
            nn.fit(z_normal)
            d, _ = nn.kneighbors(z_event)
            novelty = np.mean(d, axis=1)
        event_c["novelty_to_normal"] = novelty

        cat_base = baseline_event[baseline_event["category"] == cat]
        excess_map = {r.event_date: float(r.positive_excess) for r in cat_base.itertuples()}
        pool = select_event_seed_pool(
            event_c,
            excess_map,
            "novelty_to_normal",
            pm.per_day_seed_quota_mode,
            pm.per_day_seed_percent,
            pm.per_day_seed_fixed,
            pm.min_total_seeds_per_category,
            pm.max_total_seeds_per_category,
        )
        if pool.empty:
            continue
        pool["category"] = cat
        seed_rows.append(pool)

        seed_idx = pool.index.to_numpy()
        rel_pos = event_c.index.get_indexer(seed_idx)
        z_seed = z_event[rel_pos]
        labels = cluster_seed_pool(pool, z_seed, pm.cluster_method, pm.min_cluster_size, pm.max_clusters_per_category)
        pool = pool.copy()
        pool["cluster_id"] = labels

        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x = tfidf.fit_transform(pool["text_clean"]) if len(pool) > 1 else None

        cat_clusters = []
        valid_cluster_ids = sorted([int(v) for v in pool["cluster_id"].unique() if int(v) != -1])
        for cid in valid_cluster_ids:
            cm = pool[pool["cluster_id"] == cid]
            cm = cm.sort_values("novelty_to_normal", ascending=False)
            rel = pool.index.get_indexer(cm.index)
            centroid = z_seed[rel].mean(axis=0).tolist()
            top_terms: list[str] = []
            if x is not None and len(cm) > 0:
                rows = pool.index.get_indexer(cm.index)
                mean_vec = np.asarray(x[rows].mean(axis=0)).reshape(-1)
                feat = np.asarray(tfidf.get_feature_names_out())
                top_terms = feat[np.argsort(mean_vec)[::-1][:15]].tolist()
            cat_clusters.append(
                {
                    "cluster_id": cid,
                    "size": int(len(cm)),
                    "avg_novelty_to_normal": float(cm["novelty_to_normal"].mean()),
                    "top_terms": top_terms,
                    "centroid": centroid,
                    "representative_row_ids": cm.get("row_id", pd.Series([], dtype=object)).head(10).astype(str).tolist(),
                }
            )
            for r in cm.itertuples():
                cluster_members.append(
                    {
                        "row_id": getattr(r, "row_id", str(r.Index)),
                        "event_time": r.event_time,
                        "category": cat,
                        "cluster_id": cid,
                        "novelty_to_normal": float(r.novelty_to_normal),
                        "text_clean": r.text_clean,
                        "text_original": r.text_original,
                    }
                )

        fit_categories[cat] = {
            "vectorizer": vs["vectorizer"],
            "svd": vs["svd"],
            "z_normal": z_normal,
            "normal_centroid": np.mean(z_normal, axis=0).tolist(),
            "event_centroids": [c["centroid"] for c in cat_clusters],
            "clusters": cat_clusters,
        }
        cluster_profiles.append(
            {
                "category": cat,
                "normal_count": int(len(normal_c)),
                "event_count": int(len(event_c)),
                "seed_count": int(len(pool)),
                "clusters": cat_clusters,
            }
        )

    seed_pool_df = pd.concat(seed_rows, ignore_index=True) if seed_rows else pd.DataFrame(columns=["row_id", "event_time", "category", "text_original", "text_clean", "novelty_to_normal"])
    if not seed_pool_df.empty:
        seed_pool_df["source_day"] = pd.to_datetime(seed_pool_df["event_time"], errors="coerce").dt.date.astype(str)
        seed_pool_df["source_period"] = "event"
    seed_pool_df.to_parquet(paths.root / "seed_pool.parquet", index=False)

    cluster_members_df = pd.DataFrame(cluster_members)
    cluster_members_df.to_parquet(paths.root / "cluster_members.parquet", index=False)
    (paths.root / "cluster_profiles.json").write_text(json.dumps(cluster_profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    fit_bundle = {
        "tag": tag,
        "label_source": effective_label_source,
        "requested_label_source": label_source,
        "normal_period": normal_period,
        "event_period": event_period,
        "baseline_params": {"shrink_k": pm.baseline_weekday_shrink_k, "min_std": pm.baseline_min_std},
        "baseline_stats": {
            "global": counts_normal.groupby("category")["actual_count"].agg(global_mean="mean", global_std="std").reset_index(),
            "weekday": counts_normal.groupby(["category", "weekday"])["actual_count"].agg(weekday_mean="mean", weekday_std="std", n_cw="count").reset_index(),
        },
        "growth_summary": growth,
        "categories": fit_categories,
        "scoring": {
            "score_w_event_similarity": pm.score_w_event_similarity,
            "score_w_normal_distance": pm.score_w_normal_distance,
            "score_w_category_anomaly": pm.score_w_category_anomaly,
            "row_alert_threshold": pm.row_alert_threshold,
        },
    }
    joblib.dump(fit_bundle, paths.fit_bundle)

    fit_meta = {
        "created_at": datetime.utcnow().isoformat(),
        "tag": tag,
        "normal_period": normal_period,
        "event_period": event_period,
        "label_source": effective_label_source,
        "requested_label_source": label_source,
        "sizes": {"normal_rows": int(len(normal)), "event_rows": int(len(event)), "candidate_categories": len(candidates)},
    }
    paths.fit_meta.write_text(json.dumps(fit_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(paths.export) as writer:
        growth.to_excel(writer, sheet_name="growth_summary", index=False)
        seed_pool_df.to_excel(writer, sheet_name="seed_pool", index=False)
        cluster_members_df.to_excel(writer, sheet_name="cluster_members", index=False)

    render_pattern_fit_report(
        paths.report,
        {
            "tag": tag,
            "normal_period": normal_period,
            "event_period": event_period,
            "growth_rows": growth.head(20).to_dict(orient="records"),
            "clusters": cluster_profiles,
            "seed_examples": seed_pool_df.head(30).to_dict(orient="records"),
        },
    )

    return paths.report, paths.fit_bundle, paths.growth_summary
