from __future__ import annotations

from statistics import mean
from typing import Any


def _is_true(verdict: str) -> bool:
    return verdict == "true"


def _normalize_reviewed(rows: list[dict[str, Any]], include_uncertain_as: str = "ignore") -> list[dict[str, Any]]:
    reviewed = [r for r in rows if r.get("verdict") in {"true", "false", "uncertain"}]
    if include_uncertain_as == "false":
        return reviewed
    if include_uncertain_as == "ignore":
        return [r for r in reviewed if r.get("verdict") != "uncertain"]
    return reviewed


def compute_precision_at_k(rows: list[dict[str, Any]], k: int, score_column: str, verdict_column: str = "verdict") -> float | None:
    scored = [r for r in rows if r.get(score_column) is not None]
    if not scored:
        return None
    ranked = sorted(scored, key=lambda x: float(x.get(score_column) or 0.0), reverse=True)
    top = ranked[:k]
    if not top:
        return None
    positives = sum(1 for r in top if _is_true(str(r.get(verdict_column))))
    denom = min(k, len(ranked))
    return positives / denom if denom else None


def compute_precision_curve(rows: list[dict[str, Any]], ks: list[int], score_column: str, verdict_column: str = "verdict") -> list[dict[str, Any]]:
    return [
        {
            "k": k,
            "precision": compute_precision_at_k(rows, k, score_column=score_column, verdict_column=verdict_column),
        }
        for k in ks
    ]


def compute_precision_by_bucket(rows: list[dict[str, Any]], score_column: str) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for i in range(10):
        low = i / 10
        high = (i + 1) / 10
        label = f"{low:.1f}-{high:.1f}"
        bucket_rows = []
        for r in rows:
            score = r.get(score_column)
            if score is None:
                continue
            score_f = float(score)
            if (score_f >= low and score_f < high) or (i == 9 and score_f <= high):
                bucket_rows.append(r)
        reviewed_count = len(bucket_rows)
        true_count = sum(1 for r in bucket_rows if _is_true(str(r.get("verdict"))))
        buckets.append(
            {
                "bucket": label,
                "reviewed_count": reviewed_count,
                "true_count": true_count,
                "precision": (true_count / reviewed_count) if reviewed_count else None,
            }
        )
    return buckets


def compute_precision_by_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        group_name = str(r.get(key) or "UNKNOWN")
        item = grouped.setdefault(group_name, {"name": group_name, "reviewed_count": 0, "true_count": 0, "false_count": 0})
        item["reviewed_count"] += 1
        if r.get("verdict") == "true":
            item["true_count"] += 1
        if r.get("verdict") == "false":
            item["false_count"] += 1

    out = []
    for item in grouped.values():
        reviewed = item["reviewed_count"]
        out.append(
            {
                **item,
                "precision": (item["true_count"] / reviewed) if reviewed else None,
            }
        )
    out.sort(key=lambda x: x["reviewed_count"], reverse=True)
    return out


def compute_mode_metrics(rows: list[dict[str, Any]], mode: str, include_uncertain_as: str = "ignore") -> dict[str, Any]:
    score_column = "base_score"
    if mode in {"calibrated", "reranked"}:
        score_column = "rerank_score"

    filtered = _normalize_reviewed(rows, include_uncertain_as=include_uncertain_as)
    scored = [r for r in filtered if r.get(score_column) is not None]
    reviewed_rows = len(scored)
    true_count = sum(1 for r in scored if r.get("verdict") == "true")
    false_count = sum(1 for r in scored if r.get("verdict") == "false")
    uncertain_count = sum(1 for r in scored if r.get("verdict") == "uncertain")

    true_scores = [float(r[score_column]) for r in scored if r.get("verdict") == "true" and r.get(score_column) is not None]
    false_scores = [float(r[score_column]) for r in scored if r.get("verdict") == "false" and r.get(score_column) is not None]

    return {
        "mode": mode,
        "score_column": score_column,
        "reviewed_rows": reviewed_rows,
        "true_count": true_count,
        "false_count": false_count,
        "uncertain_count": uncertain_count,
        "precision_reviewed": (true_count / reviewed_rows) if reviewed_rows else None,
        "precision_at": compute_precision_curve(scored, [10, 20, 50, 100], score_column=score_column),
        "average_score_true": mean(true_scores) if true_scores else None,
        "average_score_false": mean(false_scores) if false_scores else None,
        "by_score_bucket": compute_precision_by_bucket(scored, score_column=score_column),
        "by_category": compute_precision_by_group(scored, key="category"),
        "by_cluster": compute_precision_by_group(scored, key="cluster"),
    }
