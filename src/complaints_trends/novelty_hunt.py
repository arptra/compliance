from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .config import ProjectConfig
from .features import TextVectorizers
from .gigachat_mtls import GigaChatNormalizer
from .reports.render import render_template
from .text_cleaning import clean_for_model, load_tokens


def _parse_month_range(month_range: str) -> tuple[str, str]:
    if ".." not in month_range:
        raise ValueError("baseline_range must be in form YYYY-MM..YYYY-MM")
    start, end = [x.strip() for x in month_range.split("..", 1)]
    if not start or not end:
        raise ValueError("baseline_range must include both start and end month")
    return start, end


def _extract_text(df: pd.DataFrame, text_field: str, use_first_message_only: bool) -> pd.Series:
    if use_first_message_only and "client_first_message" in df.columns:
        return df["client_first_message"].fillna("").astype(str)
    if text_field in df.columns:
        return df[text_field].fillna("").astype(str)
    if "client_first_message" in df.columns:
        return df["client_first_message"].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def _strip_system_markers(text: str) -> str:
    out = str(text)
    for marker in ["CHATBOT:", "OPERATOR:", "CLIENT:"]:
        out = out.replace(marker, " ")
    return out


def _clean_texts(df: pd.DataFrame, cfg: ProjectConfig) -> pd.Series:
    raw = _extract_text(df, cfg.analysis.novelty_hunt.text_field, cfg.analysis.novelty_hunt.use_first_message_only)
    if cfg.analysis.novelty_hunt.strip_system_speakers:
        raw = raw.map(_strip_system_markers)
    deny = load_tokens(cfg.files.deny_tokens_path)
    stop = load_tokens(cfg.files.extra_stopwords_path)
    deny_tokens = deny | stop
    return raw.map(lambda x: clean_for_model(x, deny_tokens))


def _select_label(df: pd.DataFrame, source: str) -> pd.Series:
    if source == "llm" and "complaint_category_llm" in df.columns:
        return df["complaint_category_llm"].fillna("OTHER").astype(str)
    if "category_pred" in df.columns:
        return df["category_pred"].fillna("OTHER").astype(str)
    if "complaint_category_llm" in df.columns:
        return df["complaint_category_llm"].fillna("OTHER").astype(str)
    return pd.Series(["OTHER"] * len(df), index=df.index, dtype=object)


def _project_space(x_base, x_new, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    max_components = max(2, min(int(n_components), x_base.shape[1] - 1, x_base.shape[0] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    z_base = svd.fit_transform(x_base)
    z_new = svd.transform(x_new)
    return z_base, z_new


def _knn_novelty(z_base: np.ndarray, z_new: np.ndarray, k: int) -> np.ndarray:
    k_eff = max(1, min(k, len(z_base)))
    nn = NearestNeighbors(metric="cosine", n_neighbors=k_eff)
    nn.fit(z_base)
    d, _ = nn.kneighbors(z_new)
    return np.mean(d, axis=1)


def _kmeans_distance(z_base: np.ndarray, z_new: np.ndarray, kmeans_k: int) -> np.ndarray:
    k = max(1, min(int(kmeans_k), len(z_base)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(z_base)
    return np.min(km.transform(z_new), axis=1)


def _lof_score(z_base: np.ndarray, z_new: np.ndarray) -> np.ndarray:
    n_neighbors = max(2, min(20, len(z_base) - 1))
    lof = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)
    lof.fit(z_base)
    return -lof.decision_function(z_new)


def _method_score(method: str, z_base: np.ndarray, z_new: np.ndarray, cfg) -> np.ndarray:
    if method == "knn_cosine":
        return _knn_novelty(z_base, z_new, cfg.knn_k)
    if method == "lof":
        return _lof_score(z_base, z_new)
    return _kmeans_distance(z_base, z_new, cfg.kmeans_k)


def _combine_scores(primary: np.ndarray, secondary: np.ndarray | None, cfg) -> np.ndarray:
    if secondary is None:
        return primary
    if cfg.combine == "mean":
        return (primary + secondary) / 2.0
    if cfg.combine == "weighted":
        return cfg.weight_primary * primary + cfg.weight_secondary * secondary
    return np.maximum(primary, secondary)


def _baseline_self_scores(z_base: np.ndarray, cfg) -> np.ndarray:
    if len(z_base) <= 2:
        return np.array([0.0])
    n = len(z_base)
    sample_size = min(2000, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=sample_size, replace=False)
    sample = z_base[idx]
    k_eff = max(2, min(cfg.knn_k + 1, n))
    nn = NearestNeighbors(metric="cosine", n_neighbors=k_eff)
    nn.fit(z_base)
    dist, neigh = nn.kneighbors(sample)
    out = []
    for i in range(len(sample)):
        row_d = dist[i]
        row_n = neigh[i]
        own = idx[i]
        filt = row_d[row_n != own]
        if len(filt) == 0:
            filt = row_d[1:]
        out.append(float(np.mean(filt)) if len(filt) else 0.0)
    return np.array(out)


def _cluster_novel(z_novel: np.ndarray, method: str, min_cluster_size: int, max_clusters: int) -> np.ndarray:
    if len(z_novel) < max(3, min_cluster_size):
        return np.full(len(z_novel), -1, dtype=int)
    if method == "dbscan":
        model = DBSCAN(eps=0.8, min_samples=max(3, min_cluster_size // 2))
        return model.fit_predict(z_novel)
    if method == "agglomerative":
        k = max(2, min(max_clusters, len(z_novel) // max(2, min_cluster_size)))
        return AgglomerativeClustering(n_clusters=k).fit_predict(z_novel)
    model = OPTICS(min_samples=max(3, min_cluster_size // 2), min_cluster_size=max(2, min_cluster_size))
    return model.fit_predict(z_novel)


def _cluster_explain(state_df: pd.DataFrame, cfg) -> list[dict[str, Any]]:
    novel = state_df[state_df["is_novel"]].copy().reset_index(drop=True)
    if novel.empty:
        return []
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    try:
        x = tfidf.fit_transform(novel["text_clean"].fillna(""))
        terms = np.array(tfidf.get_feature_names_out())
    except ValueError:
        x = None
        terms = np.array([], dtype=object)

    summaries: list[dict[str, Any]] = []
    for cid, grp in novel.groupby("cluster_id"):
        if int(cid) == -1:
            continue
        top_terms: list[str] = []
        if x is not None and len(terms) > 0:
            idx = grp.index.to_numpy(dtype=int)
            mean_vec = np.asarray(x[idx].mean(axis=0)).ravel()
            top_idx = np.argsort(mean_vec)[::-1][:15]
            top_terms = [str(terms[i]) for i in top_idx if mean_vec[i] > 0]

        ex_cols = ["row_id", "event_time", "text_original", "text_clean", "novelty_score", "category_pred"]
        ex_cols = [c for c in ex_cols if c in grp.columns]
        examples = (
            grp.sort_values("novelty_score", ascending=False)
            .head(cfg.examples_per_cluster)[ex_cols]
            .to_dict(orient="records")
        )
        cat_dist = grp["category_pred"].fillna("OTHER").astype(str).value_counts().to_dict() if "category_pred" in grp.columns else {}
        summaries.append(
            {
                "cluster_id": int(cid),
                "size": int(len(grp)),
                "avg_novelty_score": float(grp["novelty_score"].mean()),
                "category_distribution": cat_dist,
                "top_terms": top_terms,
                "examples": examples,
            }
        )
    summaries.sort(key=lambda x: (x["size"] * x["avg_novelty_score"]), reverse=True)
    return summaries


def _llm_cluster_summaries(cfg: ProjectConfig, tag: str, cluster_summaries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if not cluster_summaries:
        return {}
    if not cfg.llm.enabled:
        return {}

    norm = GigaChatNormalizer(cfg.llm, taxonomy={"category_codes": ["OTHER"], "subcategories_by_category": {}}, mock=False)
    if norm.client is None:
        return {}

    max_clusters = cfg.analysis.novelty_hunt.llm_max_clusters
    ex_per_cluster = cfg.analysis.novelty_hunt.llm_examples_per_cluster
    selected = cluster_summaries[:max_clusters]
    out: dict[int, dict[str, Any]] = {}
    for item in selected:
        cluster_id = int(item["cluster_id"])
        prompt_obj = {
            "task": "novelty_cluster_summary",
            "cluster_id": cluster_id,
            "goal": "кратко описать новый подтип/паттерн жалоб и чем он отличается от типичных формулировок",
            "return": {
                "title_ru": "3-7 слов",
                "is_consistent_cluster": True,
                "distinctive_signals": ["3-7 пунктов: отличительные детали/контекст/формулировки"],
                "suggested_subcategory_code": "snake_case_ascii_short или null",
                "suggested_subcategory_label_ru": "если есть",
                "suggested_keywords": ["5-10 слов/фраз"],
            },
            "inputs": {
                "top_terms": item.get("top_terms", [])[:15],
                "examples": [
                    {
                        "row_id": e.get("row_id"),
                        "text": e.get("text_original") or e.get("text_clean") or "",
                        "score": e.get("novelty_score"),
                    }
                    for e in item.get("examples", [])[:ex_per_cluster]
                ],
            },
        }
        k_raw = f"novelty_cluster_summary|{tag}|{cluster_id}|{json.dumps(prompt_obj, sort_keys=True, ensure_ascii=False)}"
        cache_key = hashlib.sha256(k_raw.encode("utf-8")).hexdigest()
        cached = norm.cache.get(cache_key)
        if cached:
            out[cluster_id] = cached
            continue
        response = norm.client.chat(
            {
                "model": cfg.llm.model,
                "messages": [
                    {"role": "system", "content": "Верни только валидный JSON без markdown и комментариев."},
                    {"role": "user", "content": json.dumps(prompt_obj, ensure_ascii=False)},
                ],
            }
        )
        parsed = json.loads(response.choices[0].message.content)
        norm.cache.set(cache_key, parsed)
        out[cluster_id] = parsed
    return out


def novelty_hunt(cfg: ProjectConfig, new_month: str, baseline_range: str, tag: str, use_llm_summary: bool | None = None) -> tuple[Path, Path, Path]:
    hunt_cfg = cfg.analysis.novelty_hunt
    start, end = _parse_month_range(baseline_range)

    baseline_df = pd.read_parquet(cfg.prepare.output_parquet)
    new_path = Path(hunt_cfg.interim_dir) / f"month_{new_month}.parquet"
    if not new_path.exists():
        raise FileNotFoundError(f"new month parquet not found: {new_path}")
    new_df = pd.read_parquet(new_path)

    baseline_df = baseline_df[(baseline_df["month"] >= start) & (baseline_df["month"] <= end)].copy()
    baseline_df["category_label"] = _select_label(baseline_df, hunt_cfg.label_source)
    new_df["category_label"] = _select_label(new_df, "pred")

    baseline_df["text_original"] = _extract_text(baseline_df, hunt_cfg.text_field, hunt_cfg.use_first_message_only)
    baseline_df["text_clean"] = _clean_texts(baseline_df, cfg)
    new_df["text_original"] = _extract_text(new_df, hunt_cfg.text_field, hunt_cfg.use_first_message_only)
    new_df["text_clean"] = _clean_texts(new_df, cfg)

    if hunt_cfg.candidate_pool == "complaints_only":
        candidate_mask = new_df.get("is_complaint_pred", False) == True
    elif hunt_cfg.candidate_pool == "all":
        candidate_mask = pd.Series(True, index=new_df.index)
    else:
        score = pd.to_numeric(new_df.get("complaint_score", np.nan), errors="coerce")
        candidate_mask = ((score >= hunt_cfg.complaint_score_min) & (score <= hunt_cfg.complaint_score_max)) | (new_df.get("is_complaint_pred", False) == True)

    complaint_margin_raw = new_df["complaint_margin"] if "complaint_margin" in new_df.columns else pd.Series(np.nan, index=new_df.index)
    category_margin_raw = new_df["category_margin"] if "category_margin" in new_df.columns else pd.Series(np.nan, index=new_df.index)
    complaint_margin = pd.to_numeric(complaint_margin_raw, errors="coerce")
    category_margin = pd.to_numeric(category_margin_raw, errors="coerce")
    low_margin_mask = (complaint_margin < 0.1) | (category_margin < 0.15)
    if hunt_cfg.include_low_margin and low_margin_mask.notna().any():
        candidate_mask = candidate_mask | low_margin_mask.fillna(False)
    if hunt_cfg.include_other_category and "category_pred" in new_df.columns:
        candidate_mask = candidate_mask | (new_df["category_pred"].fillna("OTHER").astype(str) == "OTHER")

    candidate_df = new_df[candidate_mask.fillna(False)].copy()
    if candidate_df.empty:
        raise ValueError("No candidate rows selected for novelty-hunt")

    baseline_text = baseline_df["text_clean"].fillna("")
    candidate_text = candidate_df["text_clean"].fillna("")

    if hunt_cfg.vectorizer_source == "trained":
        vectorizer = joblib.load(Path(cfg.training.model_dir) / "vectorizers.joblib")
    else:
        vectorizer = TextVectorizers(cfg.training)
        vectorizer.fit_transform(baseline_text)

    x_base = vectorizer.transform(baseline_text)
    x_new = vectorizer.transform(candidate_text)
    z_base, z_new = _project_space(x_base, x_new, hunt_cfg.svd_components)

    global_primary = _method_score(hunt_cfg.method_primary, z_base, z_new, hunt_cfg)
    global_secondary = None if hunt_cfg.method_secondary == "none" else _method_score(hunt_cfg.method_secondary, z_base, z_new, hunt_cfg)
    global_score = _combine_scores(global_primary, global_secondary, hunt_cfg)

    per_cat = np.zeros(len(candidate_df), dtype=float)
    base_sizes = np.zeros(len(candidate_df), dtype=int)
    cat_values = candidate_df["category_label"].fillna("OTHER").astype(str).to_numpy()
    base_cat_values = baseline_df["category_label"].fillna("OTHER").astype(str).to_numpy()
    for cat in np.unique(cat_values):
        idx_new = np.where(cat_values == cat)[0]
        idx_base = np.where(base_cat_values == cat)[0]
        base_sizes[idx_new] = len(idx_base)
        if len(idx_base) < hunt_cfg.min_baseline_per_category:
            score = global_score[idx_new]
            base_sizes[idx_new] = len(z_base)
        else:
            z_base_c = z_base[idx_base]
            p = _method_score(hunt_cfg.method_primary, z_base_c, z_new[idx_new], hunt_cfg)
            s = None if hunt_cfg.method_secondary == "none" else _method_score(hunt_cfg.method_secondary, z_base_c, z_new[idx_new], hunt_cfg)
            score = _combine_scores(p, s, hunt_cfg)
        per_cat[idx_new] = score

    if hunt_cfg.novelty_scope == "global":
        novelty_score = global_score
        scope_used = "global"
    elif hunt_cfg.novelty_scope == "per_category":
        novelty_score = per_cat
        scope_used = "per_category"
    else:
        novelty_score = np.maximum(global_score, per_cat)
        scope_used = "both"

    if hunt_cfg.select_mode == "top_k":
        k = min(hunt_cfg.top_k, len(candidate_df))
        order = np.argsort(novelty_score)[::-1]
        is_novel = np.zeros(len(candidate_df), dtype=bool)
        is_novel[order[:k]] = True
        threshold = float(novelty_score[order[k - 1]]) if k > 0 else 0.0
    elif hunt_cfg.select_mode == "threshold":
        threshold = float(hunt_cfg.threshold_value)
        is_novel = novelty_score > threshold
    else:
        baseline_self = _baseline_self_scores(z_base, hunt_cfg)
        threshold = float(np.percentile(baseline_self, hunt_cfg.threshold_percentile))
        is_novel = novelty_score > threshold

    cluster_ids = np.full(len(candidate_df), -1, dtype=int)
    novel_idx = np.where(is_novel)[0]
    if len(novel_idx) > 0:
        cluster_ids[novel_idx] = _cluster_novel(
            z_new[novel_idx],
            hunt_cfg.clustering,
            hunt_cfg.min_cluster_size,
            hunt_cfg.max_clusters,
        )

    state = candidate_df.copy()
    state["global_novelty_score"] = global_score
    state["per_category_novelty_score"] = per_cat
    state["novelty_score"] = novelty_score
    state["is_novel"] = is_novel
    state["cluster_id"] = cluster_ids
    state["novelty_scope_used"] = scope_used
    state["baseline_category_size_used"] = base_sizes
    state["complaint_margin"] = complaint_margin.reindex(state.index)
    state["category_margin"] = category_margin.reindex(state.index)

    keep_cols = [
        "row_id", "event_time", "month", "complaint_score", "is_complaint_pred", "category_pred", "subcategory_pred",
        "text_original", "text_clean", "global_novelty_score", "per_category_novelty_score", "novelty_score", "is_novel",
        "cluster_id", "complaint_margin", "category_margin", "novelty_scope_used", "baseline_category_size_used",
    ]
    keep_cols = [c for c in keep_cols if c in state.columns]
    state = state[keep_cols].copy()

    cluster_summaries = _cluster_explain(state, hunt_cfg)

    llm_enabled = hunt_cfg.llm_summary_enabled if use_llm_summary is None else use_llm_summary
    llm_summaries: dict[int, dict[str, Any]] = {}
    if llm_enabled:
        llm_summaries = _llm_cluster_summaries(cfg, tag, cluster_summaries)

    for item in cluster_summaries:
        item["llm_summary"] = llm_summaries.get(item["cluster_id"])

    interim_dir = Path(hunt_cfg.interim_dir)
    exports_dir = Path(hunt_cfg.exports_dir)
    reports_dir = Path(hunt_cfg.reports_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    state_path = interim_dir / f"novelty_hunt_state_{tag}.parquet"
    meta_path = interim_dir / f"novelty_hunt_meta_{tag}.json"
    summary_json = interim_dir / f"novelty_hunt_clusters_{tag}.json"
    llm_json = interim_dir / f"novelty_hunt_llm_{tag}.json"
    export_path = exports_dir / f"novelty_hunt_{tag}.xlsx"
    report_path = reports_dir / f"novelty_hunt_{tag}.html"

    state.to_parquet(state_path, index=False)

    meta = {
        "new_month": new_month,
        "baseline_range": baseline_range,
        "candidate_rows": int(len(candidate_df)),
        "novel_rows": int(state["is_novel"].sum()),
        "threshold": threshold,
        "scope": scope_used,
        "select_mode": hunt_cfg.select_mode,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(cluster_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if llm_summaries:
        llm_json.write_text(json.dumps(llm_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    novel_rows = state[state["is_novel"]].sort_values("novelty_score", ascending=False)
    summary_df = pd.DataFrame(
        [
            {
                "cluster_id": x["cluster_id"],
                "size": x["size"],
                "avg_novelty_score": x["avg_novelty_score"],
                "category_distribution": json.dumps(x["category_distribution"], ensure_ascii=False),
                "top_terms": ", ".join(x["top_terms"][:15]),
                "title_ru": (x.get("llm_summary") or {}).get("title_ru"),
            }
            for x in cluster_summaries
        ]
    )
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        novel_rows.to_excel(writer, sheet_name="novel_rows", index=False)
        summary_df.to_excel(writer, sheet_name="clusters_summary", index=False)
        for x in cluster_summaries[:20]:
            sid = f"samples_{x['cluster_id']}"[:31]
            pd.DataFrame(x["examples"]).to_excel(writer, sheet_name=sid, index=False)

    render_template(
        "novelty_hunt_report.html.j2",
        str(report_path),
        {
            "meta": meta,
            "cluster_summaries": cluster_summaries,
            "novel_examples": novel_rows.head(120).to_dict(orient="records"),
        },
    )
    return report_path, state_path, export_path
