from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class VizPaths:
    tag: str

    @property
    def state_parquet(self) -> Path:
        return Path("data/interim") / f"viz_state_{self.tag}.parquet"

    @property
    def meta_json(self) -> Path:
        return Path("data/interim") / f"viz_meta_{self.tag}.json"

    @property
    def report_dir(self) -> Path:
        return Path("reports") / f"viz_{self.tag}"


def save_meta(path: Path, meta: dict) -> None:
    payload = {**meta, "created_at": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_viz_state(df: pd.DataFrame, label_source: str, freq: str = "D", top_n: int = 12) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "month",
                "category",
                "subcategory",
                "metric_count",
                "metric_share",
                "metric_share_of_all",
                "label_source",
                "is_complaint_flag",
            ]
        )

    data = df.copy()
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    data = data[data["event_time"].notna()].copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "month",
                "category",
                "subcategory",
                "metric_count",
                "metric_share",
                "metric_share_of_all",
                "label_source",
                "is_complaint_flag",
            ]
        )

    data["date"] = data["event_time"].dt.to_period(freq).dt.to_timestamp()
    if "month" not in data.columns:
        data["month"] = data["event_time"].dt.strftime("%Y-%m")

    complaints = data[data["is_complaint_flag"] == True].copy()
    top_categories = complaints["category"].astype(str).value_counts().head(top_n).index.tolist()
    complaints["category"] = complaints["category"].astype(str).where(complaints["category"].astype(str).isin(top_categories), "OTHER")

    grp_complaints = (
        complaints.groupby(["date", "month", "category", "subcategory"], dropna=False)
        .size()
        .reset_index(name="metric_count")
    )
    complaints_total = grp_complaints.groupby("date")["metric_count"].sum().rename("complaints_total")

    total_all = (
        data.groupby("date")
        .size()
        .rename("all_total")
        .reset_index()
    )

    out = grp_complaints.merge(complaints_total.reset_index(), on="date", how="left").merge(total_all, on="date", how="left")
    out["metric_share"] = (out["metric_count"] / out["complaints_total"]).fillna(0.0)
    out["metric_share_of_all"] = (out["metric_count"] / out["all_total"]).fillna(0.0)
    out["label_source"] = label_source
    out["is_complaint_flag"] = True

    target_cols = [
        "date", "month", "category", "subcategory", "metric_count",
        "metric_share", "metric_share_of_all", "label_source", "is_complaint_flag",
    ]

    non_complaints = (
        data[data["is_complaint_flag"] == False]
        .groupby(["date", "month"], dropna=False)
        .size()
        .reset_index(name="metric_count")
    )
    if len(non_complaints):
        non_complaints = non_complaints.merge(total_all, on="date", how="left")
        non_complaints["category"] = "NOT_COMPLAINT"
        non_complaints["subcategory"] = "NOT_COMPLAINT"
        non_complaints["metric_share"] = 0.0
        non_complaints["metric_share_of_all"] = (non_complaints["metric_count"] / non_complaints["all_total"]).fillna(0.0)
        non_complaints["label_source"] = label_source
        non_complaints["is_complaint_flag"] = False
        non_complaints = non_complaints[target_cols]
        out = pd.concat([out[target_cols], non_complaints], ignore_index=True)
    else:
        out = out[target_cols]

    return out.sort_values(["date", "is_complaint_flag", "metric_count"], ascending=[True, False, False]).reset_index(drop=True)
