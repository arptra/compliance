from __future__ import annotations

from datetime import datetime

from ..schemas import ReportRequest, ReportResponse
from .overview_service import OverviewService
from .pattern_monitor_service import PatternMonitorService
from .timeseries_service import (
    aggregate_timeseries_overall,
    compute_category_contribution,
    filter_by_date,
    resolve_compare_window,
)
from .data_loader import DataLoader


class ReportService:
    def __init__(self, overview: OverviewService, monitor: PatternMonitorService, loader: DataLoader) -> None:
        self.overview = overview
        self.monitor = monitor
        self.loader = loader

    def build_report(self, report_type: str, req: ReportRequest) -> ReportResponse:
        filters = req.filters
        overview = self.overview.get_overview(filters)
        monitor = self.monitor.summary(filters.get("pattern_tag", "latest"), filters)

        df = self.loader.load_prepare()
        col = "date" if "date" in df.columns else "event_time"
        actual = filter_by_date(df, col, filters.get("date_from"), filters.get("date_to"))
        b_from, b_to = resolve_compare_window(
            filters.get("date_from"),
            filters.get("date_to"),
            filters.get("baseline_mode", "previous_period"),
            filters.get("baseline_date_from"),
            filters.get("baseline_date_to"),
        )
        baseline = filter_by_date(df, col, b_from.isoformat() if b_from is not None else None, b_to.isoformat() if b_to is not None else None)
        ts = aggregate_timeseries_overall(actual, baseline, col, col, filters.get("granularity", "D"))
        contrib = compute_category_contribution(actual, baseline)[:10]

        sections = [
            {"title": "Период", "body": f"Сформировано: {datetime.utcnow().isoformat()}Z"},
            {"title": "KPI", "body": [k.model_dump() for k in overview.kpis]},
            {"title": "Timeseries summary", "body": ts.get("summary", {})},
            {"title": "Top growth contribution", "body": contrib},
            {"title": "Pattern monitor", "body": monitor.summary},
        ]
        summary = overview.executive_summary
        metrics = {
            "actual_total": next((k.value for k in overview.kpis if k.key == "actual_total"), 0),
            "alert_rows": monitor.summary.get("alert_rows", 0),
            "timeseries_delta_abs": ts.get("summary", {}).get("delta_abs", 0),
        }
        md = self._to_markdown(report_type, sections)
        html = self._to_html(report_type, sections)
        return ReportResponse(
            report_type=report_type,
            sections=sections,
            text_summary=summary,
            metrics=metrics,
            examples=[],
            markdown=md if req.output_format in {"markdown", "json"} else None,
            html=html if req.output_format in {"html", "json"} else None,
        )

    def _to_markdown(self, report_type: str, sections: list[dict]) -> str:
        lines = [f"# {report_type.title()} report"]
        for s in sections:
            lines.append(f"\n## {s['title']}\n")
            lines.append(str(s["body"]))
        return "\n".join(lines)

    def _to_html(self, report_type: str, sections: list[dict]) -> str:
        body = "".join(f"<section><h2>{s['title']}</h2><pre>{s['body']}</pre></section>" for s in sections)
        return f"<html><body><h1>{report_type.title()} report</h1>{body}</body></html>"
