from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import CheckButtons, RadioButtons, RangeSlider


def run_viewer(state_parquet: str | Path, tag: str | None = None) -> None:
    state = pd.read_parquet(state_parquet)
    if state.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет данных для визуализации", ha="center", va="center")
        ax.axis("off")
        plt.show()
        return

    d = state[state["is_complaint_flag"] == True].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].copy()
    categories = sorted(d["category"].astype(str).unique().tolist())
    active = {c: True for c in categories}
    metric_mode = {"value": "metric_count"}

    dmin = float(d["date"].min().value)
    dmax = float(d["date"].max().value)

    fig = plt.figure(figsize=(14, 8))
    ax_main = fig.add_axes([0.08, 0.22, 0.64, 0.72])
    ax_checks = fig.add_axes([0.75, 0.4, 0.22, 0.5])
    ax_slider = fig.add_axes([0.08, 0.08, 0.64, 0.05])
    ax_radio = fig.add_axes([0.75, 0.25, 0.22, 0.12])

    checks = CheckButtons(ax_checks, categories, [True] * len(categories))
    slider = RangeSlider(ax_slider, "Date range", dmin, dmax, valinit=(dmin, dmax))
    radio = RadioButtons(ax_radio, ["metric_count", "metric_share"], active=0)

    def _render():
        ax_main.clear()
        left, right = slider.val
        date_mask = (d["date"].astype("int64") >= left) & (d["date"].astype("int64") <= right)
        cats = [c for c in categories if active[c]]
        cur = d[date_mask & d["category"].isin(cats)].copy()
        if cur.empty:
            ax_main.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax_main.set_title(f"viz-view {tag or ''}")
            fig.canvas.draw_idle()
            return
        pivot = cur.pivot_table(index="date", columns="category", values=metric_mode["value"], aggfunc="sum", fill_value=0).sort_index()
        ax_main.stackplot(pivot.index, [pivot[c].values for c in pivot.columns], labels=pivot.columns)
        ax_main.set_title(f"viz-view {tag or ''} ({metric_mode['value']})")
        ax_main.legend(loc="upper left", ncol=2, fontsize=8)
        fig.canvas.draw_idle()

    def _on_check(label):
        active[label] = not active[label]
        _render()

    def _on_slider(_):
        _render()

    def _on_radio(label):
        metric_mode["value"] = label
        _render()

    checks.on_clicked(_on_check)
    slider.on_changed(_on_slider)
    radio.on_clicked(_on_radio)
    _render()
    plt.show()
