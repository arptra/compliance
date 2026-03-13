from __future__ import annotations

from datetime import datetime

from ..schemas import ReportRequest, ReportResponse
from .overview_service import OverviewService
from .pattern_monitor_service import PatternMonitorService


class ReportService:
    def __init__(self, overview: OverviewService, monitor: PatternMonitorService) -> None:
        self.overview = overview
        self.monitor = monitor

    def build_report(self, report_type: str, req: ReportRequest) -> ReportResponse:
        filters = req.filters
        overview = self.overview.get_overview(filters)
        monitor = self.monitor.summary(filters.get("pattern_tag", "latest"), filters)
        sections = [
            {"title": "Период", "body": f"Сформировано: {datetime.utcnow().isoformat()}Z"},
            {"title": "KPI", "body": [k.model_dump() for k in overview.kpis]},
            {"title": "Pattern monitor", "body": monitor.summary},
        ]
        summary = overview.executive_summary
        metrics = {
            "actual_total": next((k.value for k in overview.kpis if k.key == "actual_total"), 0),
            "alert_rows": monitor.summary.get("alert_rows", 0),
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
