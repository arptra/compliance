from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.neighbors import NearestNeighbors

from .config import ProjectConfig
from .reports.render import render_template
from .text_cleaning import clean_for_model, load_tokens

SYSTEM_MARKERS = ("CLIENT:", "OPERATOR:", "CHATBOT:")


class CombinedVectorizer:
    def __init__(self, word_vectorizer, char_vectorizer):
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer

    def transform(self, texts):
        from scipy.sparse import hstack

        return hstack([self.word_vectorizer.transform(texts), self.char_vectorizer.transform(texts)])


def parse_month_range(value: str) -> tuple[str, str]:
    if ".." not in value:
        raise ValueError("Period must be in form YYYY-MM..YYYY-MM")
    start, end = [x.strip() for x in value.split("..", 1)]
    if not start or not end:
        raise ValueError("Both start and end months are required")
    return start, end


def get_label_columns(df: pd.DataFrame, label_source: str) -> tuple[pd.Series, pd.Series]:
    if label_source == "llm":
        is_complaint = df.get("is_complaint_llm", pd.Series(False, index=df.index))
        category = df.get("complaint_category_llm", pd.Series("OTHER", index=df.index))
    else:
        is_complaint = df.get("is_complaint_pred", pd.Series(False, index=df.index))
        category = df.get("category_pred", pd.Series("OTHER", index=df.index))
    return is_complaint.fillna(False).astype(bool), category.fillna("OTHER").astype(str)


def select_period_rows(df: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = parse_month_range(period)
    months = df.get("month")
    if months is None:
        months = pd.to_datetime(df.get("event_time"), errors="coerce").dt.strftime("%Y-%m")
    return df[(months >= start) & (months <= end)].copy()


def build_text_clean(df: pd.DataFrame, cfg: ProjectConfig, text_field: str, use_first_message_only: bool, strip_system_speakers: bool) -> pd.Series:
    if use_first_message_only and "client_first_message" in df.columns:
        text = df["client_first_message"].fillna("").astype(str)
    elif text_field in df.columns:
        text = df[text_field].fillna("").astype(str)
    else:
        text = df.get("client_first_message", pd.Series("", index=df.index)).fillna("").astype(str)

    if strip_system_speakers:
        for m in SYSTEM_MARKERS:
            text = text.str.replace(m, " ", regex=False)

    deny_tokens = load_tokens(cfg.files.deny_tokens_path) | load_tokens(cfg.files.extra_stopwords_path)
    return text.map(lambda x: clean_for_model(x, deny_tokens))


def build_daily_category_counts(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    out = df.copy()
    out["event_date"] = pd.to_datetime(out["event_time"], errors="coerce").dt.date
    out = out.dropna(subset=["event_date"])
    grp = out.groupby(["event_date", category_col], as_index=False).size().rename(columns={"size": "actual_count", category_col: "category"})
    grp["weekday"] = pd.to_datetime(grp["event_date"]).dt.weekday
    return grp


def build_category_baseline(normal_counts: pd.DataFrame, target_counts: pd.DataFrame, shrink_k: float, min_std: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_global = normal_counts.groupby("category")["actual_count"].agg(global_mean="mean", global_std="std").reset_index()
    stats_weekday = normal_counts.groupby(["category", "weekday"])["actual_count"].agg(weekday_mean="mean", weekday_std="std", n_cw="count").reset_index()

    base = target_counts.merge(stats_global, on="category", how="left").merge(stats_weekday, on=["category", "weekday"], how="left")
    cols = ["global_mean", "global_std", "weekday_mean", "weekday_std", "n_cw"]
    base[cols] = base[cols].fillna(0.0)
    weight = base["n_cw"] / (base["n_cw"] + float(shrink_k))
    base["expected_count"] = weight * base["weekday_mean"] + (1.0 - weight) * base["global_mean"]

    fallback_std = np.sqrt(np.maximum(base["expected_count"], 1.0))
    std = base["weekday_std"].where(base["weekday_std"] > 0, np.nan)
    std = std.fillna(base["global_std"].where(base["global_std"] > 0, np.nan)).fillna(fallback_std)
    base["std_count"] = np.maximum(std, float(min_std))
    base["residual"] = base["actual_count"] - base["expected_count"]
    base["zscore"] = base["residual"] / base["std_count"]
    base["positive_excess"] = np.maximum(base["residual"], 0.0)
    return base, stats_weekday


def _scale(v: pd.Series) -> pd.Series:
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - lo) / (hi - lo)


def compute_growth_summary(event_baseline: pd.DataFrame, anomaly_z_threshold: float, anomaly_min_excess_total: int, anomaly_min_event_days: int, top_growth_categories: int) -> pd.DataFrame:
    agg = event_baseline.groupby("category", as_index=False).agg(
        total_actual_event=("actual_count", "sum"),
        total_expected_event=("expected_count", "sum"),
        total_excess_event=("positive_excess", "sum"),
        max_zscore=("zscore", "max"),
        mean_positive_z=("zscore", lambda s: float(np.mean(np.maximum(s, 0.0)))),
        event_days_over_threshold=("zscore", lambda s: int((s >= anomaly_z_threshold).sum())),
    )
    agg["growth_score"] = 0.45 * _scale(agg["total_excess_event"]) + 0.35 * _scale(agg["max_zscore"]) + 0.20 * _scale(agg["event_days_over_threshold"])
    top_set = set(agg.sort_values("growth_score", ascending=False).head(top_growth_categories)["category"].tolist())
    agg["is_candidate"] = (
        ((agg["total_excess_event"] >= anomaly_min_excess_total) & (agg["event_days_over_threshold"] >= anomaly_min_event_days))
        | agg["category"].isin(top_set)
    )
    return agg.sort_values(["is_candidate", "growth_score"], ascending=[False, False]).reset_index(drop=True)


def build_category_vector_space(normal_text: pd.Series, event_text: pd.Series, cfg: ProjectConfig, vectorizer_source: str, svd_components: int, random_state: int) -> dict[str, Any]:
    if vectorizer_source == "trained":
        vectorizer = joblib.load(Path(cfg.training.model_dir) / "vectorizers.joblib")
        x_normal = vectorizer.transform(normal_text)
        x_event = vectorizer.transform(event_text)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack

        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=1.0, sublinear_tf=True)
        xw_n = wv.fit_transform(normal_text)
        xc_n = cv.fit_transform(normal_text)
        x_normal = hstack([xw_n, xc_n])
        x_event = hstack([wv.transform(event_text), cv.transform(event_text)])

        vectorizer = CombinedVectorizer(wv, cv)

    from scipy.sparse import vstack

    x_fit = vstack([x_normal, x_event]) if x_event.shape[0] else x_normal
    max_components = max(2, min(int(svd_components), x_fit.shape[1] - 1, x_fit.shape[0] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    z_fit = svd.fit_transform(x_fit)
    z_normal = z_fit[: x_normal.shape[0]]
    z_event = z_fit[x_normal.shape[0] :]
    return {"vectorizer": vectorizer, "svd": svd, "z_normal": z_normal, "z_event": z_event}


def select_event_seed_pool(event_rows: pd.DataFrame, category_day_excess: dict[Any, float], novelty_col: str, per_day_seed_quota_mode: str, per_day_seed_percent: float, per_day_seed_fixed: int, min_total_seeds: int, max_total_seeds: int) -> pd.DataFrame:
    selected = []
    event_rows = event_rows.copy()
    event_rows["event_date"] = pd.to_datetime(event_rows["event_time"], errors="coerce").dt.date
    for day, block in event_rows.groupby("event_date"):
        block = block.sort_values(novelty_col, ascending=False)
        excess = category_day_excess.get(day, 0.0)
        quota = int(round(max(excess, 0.0))) if per_day_seed_quota_mode == "residual" else (int(math.ceil(len(block) * per_day_seed_percent)) if per_day_seed_quota_mode == "percent" else int(per_day_seed_fixed))
        quota = max(0, min(len(block), quota))
        if quota:
            selected.append(block.head(quota))
    pool = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=event_rows.columns)
    if len(pool) < min_total_seeds:
        pool = event_rows.sort_values(novelty_col, ascending=False).head(min_total_seeds)
    if len(pool) > max_total_seeds:
        pool = pool.sort_values(novelty_col, ascending=False).head(max_total_seeds)
    return pool


def cluster_seed_pool(seed_df: pd.DataFrame, z_seed: np.ndarray, cluster_method: str, min_cluster_size: int, max_clusters: int) -> np.ndarray:
    n = len(seed_df)
    if n == 0:
        return np.array([], dtype=int)
    if n < min_cluster_size:
        return np.zeros(n, dtype=int)
    if cluster_method == "dbscan":
        labels = DBSCAN(eps=0.7, min_samples=max(3, min_cluster_size // 2)).fit_predict(z_seed)
    elif cluster_method == "agglomerative":
        k = max(2, min(max_clusters, n // max(2, min_cluster_size)))
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(z_seed)
    else:
        labels = OPTICS(min_samples=max(3, min_cluster_size // 2), min_cluster_size=max(2, min_cluster_size)).fit_predict(z_seed)

    counts = pd.Series(labels).value_counts()
    valid = {int(k) for k, v in counts.items() if k != -1 and int(v) >= min_cluster_size}
    labels = np.array([int(v) if int(v) in valid else -1 for v in labels], dtype=int)
    kept = [int(k) for k, _ in pd.Series(labels[labels != -1]).value_counts().head(max_clusters).items()]
    labels = np.array([v if v in kept else -1 for v in labels], dtype=int)
    if (labels != -1).sum() == 0 and n > 0:
        labels[:] = 0
    return labels


def score_target_rows(target: pd.DataFrame, fit_bundle: dict[str, Any], cfg_pm) -> pd.DataFrame:
    out = target.copy()
    out["event_date"] = pd.to_datetime(out["event_time"], errors="coerce").dt.date
    out[["event_similarity", "normal_similarity", "normal_distance", "day_category_anomaly", "row_novelty_to_normal"]] = 0.0

    for cat, idx in out.groupby("category").groups.items():
        cat_info = fit_bundle.get("categories", {}).get(cat)
        if not cat_info:
            continue
        z = cat_info["svd"].transform(cat_info["vectorizer"].transform(out.loc[idx, "text_clean"]))
        n_sim = cosine_similarity(z, np.asarray(cat_info["normal_centroid"]).reshape(1, -1)).reshape(-1)
        out.loc[idx, "normal_similarity"] = n_sim
        out.loc[idx, "normal_distance"] = np.clip(1.0 - n_sim, 0.0, 1.0)

        nn = NearestNeighbors(metric="cosine", n_neighbors=max(1, min(cfg_pm.score_topk_normal_neighbors, len(cat_info["z_normal"]))))
        nn.fit(cat_info["z_normal"])
        d, _ = nn.kneighbors(z)
        out.loc[idx, "row_novelty_to_normal"] = np.mean(d, axis=1)

        centroids = np.asarray(cat_info.get("event_centroids", []))
        if centroids.size:
            sim = cosine_similarity(z, centroids)
            topk = np.sort(sim, axis=1)[:, -min(cfg_pm.score_topk_event_neighbors, sim.shape[1]) :]
            out.loc[idx, "event_similarity"] = np.clip(np.mean(topk, axis=1), 0.0, 1.0)

    day_counts = out.groupby(["event_date", "category"], as_index=False).size().rename(columns={"size": "actual_count"})
    day_counts["weekday"] = pd.to_datetime(day_counts["event_date"]).dt.weekday
    stats = fit_bundle["baseline_stats"]
    merged = day_counts.merge(stats["global"], on="category", how="left").merge(stats["weekday"], on=["category", "weekday"], how="left")
    merged[["global_mean", "weekday_mean", "weekday_std", "n_cw", "global_std"]] = merged[["global_mean", "weekday_mean", "weekday_std", "n_cw", "global_std"]].fillna(0.0)
    k, min_std = fit_bundle["baseline_params"]["shrink_k"], fit_bundle["baseline_params"]["min_std"]
    w = merged["n_cw"] / (merged["n_cw"] + k)
    merged["expected_count"] = w * merged["weekday_mean"] + (1 - w) * merged["global_mean"]
    fallback = np.sqrt(np.maximum(merged["expected_count"], 1.0))
    std = merged["weekday_std"].where(merged["weekday_std"] > 0, np.nan).fillna(merged["global_std"].where(merged["global_std"] > 0, np.nan)).fillna(fallback)
    merged["zscore"] = (merged["actual_count"] - merged["expected_count"]) / np.maximum(std, min_std)
    merged["day_category_anomaly"] = np.clip((merged["zscore"] + 1.0) / 4.0, 0.0, 1.0)
    anom = {(r.event_date, r.category): float(r.day_category_anomaly) for r in merged.itertuples()}
    out["day_category_anomaly"] = [anom.get((d, c), 0.0) for d, c in zip(out["event_date"], out["category"])]

    out["pattern_like_raw"] = cfg_pm.score_w_event_similarity * out["event_similarity"] + cfg_pm.score_w_normal_distance * out["normal_distance"] + cfg_pm.score_w_category_anomaly * out["day_category_anomaly"]
    out["pattern_like_score"] = 1.0 / (1.0 + np.exp(-4.0 * (out["pattern_like_raw"] - 0.5)))
    out["is_pattern_alert"] = out["pattern_like_score"] >= cfg_pm.row_alert_threshold
    return out


def aggregate_daily_pressure(scored: pd.DataFrame, cfg_pm, previous_state: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (day, cat), block in scored.groupby(["event_date", "category"]):
        top = block.sort_values("pattern_like_score", ascending=False).head(cfg_pm.daily_top_k_rows)
        vals = top["pattern_like_score"].to_numpy(dtype=float)
        pressure = float(1.0 - np.prod(1.0 - vals)) if cfg_pm.daily_pressure_mode == "one_minus_prod" else float(min(1.0, vals.sum()))
        rows.append({"date": day, "category": cat, "category_pressure": pressure, "rows_count": len(block)})
    cat_daily = pd.DataFrame(rows)
    if cat_daily.empty:
        empty = pd.DataFrame(columns=["date", "overall_pressure", "smoothed_state", "top_categories", "num_alert_rows", "num_categories_over_day_threshold"])
        return cat_daily, empty

    states = []
    prev = previous_state
    for day, block in cat_daily.groupby("date"):
        top = block.sort_values("category_pressure", ascending=False)
        overall = float(min(1.0, top["category_pressure"].head(5).sum()))
        smoothed = overall if prev is None else cfg_pm.state_alpha * overall + (1.0 - cfg_pm.state_alpha) * prev
        prev = smoothed
        alerts = scored[(scored["event_date"] == day) & (scored["is_pattern_alert"] == True)]
        states.append({
            "date": day,
            "overall_pressure": overall,
            "smoothed_state": smoothed,
            "top_categories": ", ".join(top.head(3)["category"].astype(str).tolist()),
            "num_alert_rows": int(len(alerts)),
            "num_categories_over_day_threshold": int((top["category_pressure"] >= cfg_pm.day_alert_threshold).sum()),
        })
    return cat_daily, pd.DataFrame(states)


def render_pattern_fit_report(path: Path, context: dict[str, Any]) -> None:
    render_template("pattern_fit_report.html.j2", str(path), context)


def render_pattern_monitor_report(path: Path, context: dict[str, Any]) -> None:
    render_template("pattern_monitor_report.html.j2", str(path), context)
