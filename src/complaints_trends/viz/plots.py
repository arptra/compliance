from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _empty_plot(path: Path, title: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path).replace("\\", "/")


def plot_stacked_area(df_counts: pd.DataFrame, out_path: Path, title: str) -> str:
    if df_counts.empty:
        return _empty_plot(out_path, title)
    p = df_counts.pivot_table(index="date", columns="category", values="metric_count", aggfunc="sum", fill_value=0).sort_index()
    if p.empty:
        return _empty_plot(out_path, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(p.index, [p[c].values for c in p.columns], labels=p.columns)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path).replace("\\", "/")


def plot_share_lines(df_share: pd.DataFrame, out_path: Path, title: str) -> str:
    if df_share.empty:
        return _empty_plot(out_path, title)
    p = df_share.pivot_table(index="date", columns="category", values="metric_share", aggfunc="sum", fill_value=0).sort_index()
    if p.empty:
        return _empty_plot(out_path, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in p.columns:
        ax.plot(p.index, p[c].values, label=str(c))
    ax.set_title(title)
    ax.set_ylabel("Share of complaints")
    ax.set_ylim(0, max(0.01, float(p.max().max()) * 1.15))
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path).replace("\\", "/")


def plot_pareto(df_period: pd.DataFrame, out_path: Path, title: str) -> str:
    if df_period.empty:
        return _empty_plot(out_path, title)
    counts = df_period.groupby("category")["metric_count"].sum().sort_values(ascending=False)
    if counts.empty:
        return _empty_plot(out_path, title)
    cum = counts.cumsum() / counts.sum()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(counts.index.astype(str), counts.values, color="#1f77b4")
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_ylabel("Count")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(cum)), cum.values, color="#d62728", marker="o")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative share")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path).replace("\\", "/")


def plot_heatmap_dow_hour(df_raw: pd.DataFrame, out_path: Path, title: str) -> str:
    if df_raw.empty or "event_time" not in df_raw.columns:
        return _empty_plot(out_path, title)
    d = df_raw.copy()
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d = d[d["event_time"].notna()]
    if d.empty:
        return _empty_plot(out_path, title)
    d["dow"] = d["event_time"].dt.dayofweek
    d["hour"] = d["event_time"].dt.hour
    hm = d.pivot_table(index="dow", columns="hour", values="event_time", aggfunc="count", fill_value=0)
    if hm.empty:
        return _empty_plot(out_path, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(hm.values, aspect="auto", origin="lower", cmap="Blues")
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(hm.index)])
    ax.set_xticks(range(0, hm.shape[1], max(1, hm.shape[1] // 8)))
    ax.set_xticklabels([str(hm.columns[i]) for i in range(0, hm.shape[1], max(1, hm.shape[1] // 8))])
    ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day of week")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path).replace("\\", "/")


def plot_delta_bars_or_waterfall(df_delta: pd.DataFrame, out_path: Path, title: str) -> str:
    if df_delta.empty:
        return _empty_plot(out_path, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    d = df_delta.sort_values("delta_pp", ascending=False).head(20)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in d["delta_pp"].values]
    ax.barh(d["category"].astype(str), d["delta_pp"].values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Delta share, pp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path).replace("\\", "/")
