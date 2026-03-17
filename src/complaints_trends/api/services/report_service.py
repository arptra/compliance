from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..schemas import (
    AlertExampleCard,
    ExecutiveCharts,
    ExecutiveExport,
    ExecutiveKpis,
    ExecutiveReportRequest,
    ExecutiveReportResponse,
    PatternRisk,
    PrimaryArea,
    ReportRequest,
    ReportResponse,
    SummaryBlock,
)
from .data_loader import DataLoader
from .overview_service import OverviewService
from .pattern_monitor_service import PatternMonitorService
from .timeseries_service import aggregate_timeseries_overall, compute_category_contribution, filter_by_date, resolve_compare_window


@dataclass
class OwnershipRecord:
    category: str
    owner_team: str | None
    owner_area: str | None


class ReportService:
    def __init__(self, overview: OverviewService, monitor: PatternMonitorService, loader: DataLoader) -> None:
        self.overview = overview
        self.monitor = monitor
        self.loader = loader

    def build_report(self, report_type: str, req: ReportRequest) -> ReportResponse:
        filters = req.filters
        overview = self.overview.get_overview(filters)
        monitor = self.monitor.summary(filters.get("pattern_tag", "latest"), filters)
        monitor_summary = monitor.summary.model_dump()

        df = self.loader.load_prepare_timeseries()
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
            {"title": "Pattern monitor", "body": monitor_summary},
            {"title": "Pattern risk", "body": {"label": monitor_summary.get("pattern_risk_label", "unavailable"), "score": monitor_summary.get("pattern_risk_score"), "calc_mode": monitor_summary.get("pattern_risk_calc_mode", "unavailable"), "alert_rows": monitor_summary.get("alert_rows", 0), "pressure_days": monitor_summary.get("pressure_days", 0)}},
        ]
        risk_phrase = self._pattern_risk_phrase(monitor_summary.get("pattern_risk_label", "unavailable"))
        summary = f"{overview.executive_summary} {risk_phrase}".strip()
        metrics = {
            "actual_total": next((k.value for k in overview.kpis if k.key == "actual_total"), 0),
            "alert_rows": monitor_summary.get("alert_rows", 0),
            "timeseries_delta_abs": ts.get("summary", {}).get("delta_abs", 0),
            "pattern_risk_score": monitor_summary.get("pattern_risk_score"),
            "pattern_risk_label": monitor_summary.get("pattern_risk_label", "unavailable"),
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

    def build_executive_report_payload(self, req: ExecutiveReportRequest) -> ExecutiveReportResponse:
        df = self.loader.load_prepare_timeseries()
        date_col = "date" if "date" in df.columns else "event_time"
        actual = filter_by_date(df, date_col, req.date_from, req.date_to)
        if req.categories:
            actual = actual[actual.get("category", pd.Series(dtype=str)).isin(req.categories)] if "category" in actual.columns else actual
        b_from, b_to = resolve_compare_window(req.date_from, req.date_to, req.compare_mode, req.baseline_date_from, req.baseline_date_to)
        baseline = filter_by_date(df, date_col, b_from.isoformat() if b_from is not None else None, b_to.isoformat() if b_to is not None else None)
        if req.categories:
            baseline = baseline[baseline.get("category", pd.Series(dtype=str)).isin(req.categories)] if "category" in baseline.columns else baseline

        ts = aggregate_timeseries_overall(actual, baseline, date_col, date_col, "D")
        contrib_rows = compute_category_contribution(actual, baseline)

        top_contrib = [r for r in contrib_rows if r["delta_abs"] > 0][:7]
        if len(top_contrib) > 6:
            tail = top_contrib[6:]
            top_contrib = top_contrib[:6] + [{
                "category": "OTHER",
                "actual": sum(x["actual_count"] for x in tail),
                "expected": sum(x["expected_count"] for x in tail),
                "delta": sum(x["delta_abs"] for x in tail),
                "contribution_pct": sum(x["contribution_to_growth"] for x in tail) * 100,
            }]
        contribution = [
            {
                "category": r["category"],
                "actual": float(r.get("actual", r.get("actual_count", 0.0))),
                "expected": float(r.get("expected", r.get("expected_count", 0.0))),
                "delta": float(r.get("delta", r.get("delta_abs", 0.0))),
                "contribution_pct": float(r.get("contribution_pct", r.get("contribution_to_growth", 0.0) * 100.0)),
            }
            for r in top_contrib
        ]

        priority_rows = []
        for row in contrib_rows:
            d_pct = row.get("delta_pct")
            priority = "high" if row.get("delta_abs", 0) > 5 else "medium" if row.get("delta_abs", 0) > 1 else "low"
            priority_rows.append(
                {
                    "category": row["category"],
                    "actual": float(row["actual_count"]),
                    "expected": float(row["expected_count"]),
                    "delta": float(row["delta_abs"]),
                    "delta_pct": (None if d_pct is None else float(d_pct * 100)),
                    "priority": priority,
                }
            )
        priority_rows = sorted(priority_rows, key=lambda x: x["delta"], reverse=True)[:7]

        monitor_summary = {}
        try:
            monitor_summary = self.monitor.summary(req.pattern_tag, {"date_from": req.date_from, "date_to": req.date_to}).summary.model_dump()
        except Exception:
            monitor_summary = {}

        risk = PatternRisk(
            score=monitor_summary.get("pattern_risk_score"),
            label=monitor_summary.get("pattern_risk_label", "unavailable"),
            display_label=monitor_summary.get("pattern_risk_display_label", "Недоступно"),
            status=monitor_summary.get("pattern_risk_status", "neutral"),
            calc_mode=monitor_summary.get("pattern_risk_calc_mode", "unavailable"),
        )

        primary_area = self._resolve_primary_area(contrib_rows, req.include_ownership)

        total = int(round(ts.get("summary", {}).get("actual_total", 0)))
        expected = int(round(ts.get("summary", {}).get("expected_total", 0)))
        delta_abs = int(round(ts.get("summary", {}).get("delta_abs", 0)))
        delta_pct = ts.get("summary", {}).get("delta_pct")
        above_baseline = sum(1 for r in contrib_rows if r.get("delta_abs", 0) > 1)
        top_growth = next((r["category"] for r in contrib_rows if r.get("delta_abs", 0) > 0), None)

        actual_expected = [
            {
                "date": p["date"],
                "actual": float(p["actual"]),
                "expected": float(p["expected"]),
                "delta": float(p["delta_abs"]),
                "delta_pct": (None if p["delta_pct"] is None else float(p["delta_pct"] * 100)),
            }
            for p in ts.get("delta", [])
        ]

        examples = self._build_examples(req, actual, contrib_rows)

        summary = self._build_summary(total, expected, delta_abs, delta_pct, contribution, risk, primary_area, examples)
        definitions = self._definitions()
        markdown = self.build_executive_markdown(summary, total, delta_abs, delta_pct, contribution, risk, primary_area)
        html = self.build_executive_html(summary, total, delta_abs, delta_pct, contribution, risk, primary_area)

        return ExecutiveReportResponse(
            meta={
                "generated_at": f"{datetime.utcnow().isoformat()}Z",
                "date_from": req.date_from,
                "date_to": req.date_to,
                "compare_mode": req.compare_mode,
                "baseline_date_from": req.baseline_date_from,
                "baseline_date_to": req.baseline_date_to,
                "categories": req.categories or [],
                "include_examples": req.include_examples,
                "include_ownership": req.include_ownership,
                "alert_rows": monitor_summary.get("alert_rows", 0),
            },
            kpis=ExecutiveKpis(
                total_complaints=total,
                expected_complaints=expected,
                delta_abs=delta_abs,
                delta_pct=(None if delta_pct is None else float(delta_pct * 100)),
                categories_above_baseline=above_baseline,
                top_growth_category=top_growth,
                pattern_risk=risk,
                pattern_risk_score=risk.score,
                pattern_risk_label=risk.label,
                pattern_risk_display_label=risk.display_label,
                pattern_risk_status=risk.status,
                primary_area=primary_area,
            ),
            charts=ExecutiveCharts(
                actual_expected=actual_expected,
                category_contribution=contribution,
                category_priority=priority_rows,
                alert_examples=examples,
            ),
            summary=summary,
            definitions=definitions,
            export=ExecutiveExport(markdown=markdown, html=html),
        )

    def _resolve_primary_area(self, contributions: list[dict], include_ownership: bool) -> PrimaryArea | None:
        if not include_ownership:
            return None
        rows = self._load_ownership_mapping()
        if not rows:
            return None
        by_category = {r["category"]: r for r in contributions}
        weights: dict[str, float] = {}
        for rec in rows:
            delta = by_category.get(rec.category, {}).get("delta_abs", 0.0)
            area = rec.owner_area or rec.owner_team
            if area and delta > 0:
                weights[area] = weights.get(area, 0.0) + float(delta)
        if not weights:
            return None
        area = max(weights.items(), key=lambda x: x[1])[0]
        return PrimaryArea(label=area, confidence_note="Наиболее вероятная зона для первичной проверки на основе вклада категорий.")

    def _load_ownership_mapping(self) -> list[OwnershipRecord]:
        candidates = [
            Path("configs/category_ownership.csv"),
            Path("data/reference/category_ownership.csv"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            required = {"category"}
            if not required.issubset(set(df.columns)):
                continue
            out: list[OwnershipRecord] = []
            for r in df.fillna("").to_dict(orient="records"):
                out.append(OwnershipRecord(category=str(r.get("category", "")), owner_team=r.get("owner_team") or None, owner_area=r.get("owner_area") or None))
            return out
        return []

    def _build_examples(self, req: ExecutiveReportRequest, actual: pd.DataFrame, contrib_rows: list[dict]) -> list[AlertExampleCard]:
        if not req.include_examples:
            return []

        def _from_df(df: pd.DataFrame) -> list[AlertExampleCard]:
            if df.empty:
                return []
            work = df.copy()
            if "category" not in work.columns:
                work["category"] = "UNKNOWN"
            text_col = next((c for c in ("text", "complaint_text", "raw_dialog", "row_dialog", "dialog_text", "client_first_message") if c in work.columns), None)
            if text_col is None:
                return []
            top_cat = [r["category"] for r in contrib_rows if r.get("delta_abs", 0) > 0][:3]
            sample = work[work["category"].isin(top_cat)] if top_cat else work
            if sample.empty:
                sample = work
            sample = sample.head(5)
            out: list[AlertExampleCard] = []
            for _, row in sample.iterrows():
                reason = "Рост категории относительно baseline"
                cat = str(row.get("category", "UNKNOWN"))
                if cat == (top_cat[0] if top_cat else ""):
                    reason = "Ключевой вклад в общий рост"
                out.append(
                    AlertExampleCard(
                        text=str(row.get(text_col, ""))[:2000],
                        category=cat,
                        reason=reason,
                        priority="high" if cat in top_cat[:1] else "medium",
                        score=None,
                    )
                )
            return out

        direct = _from_df(actual)
        if direct:
            return direct

        # fallback to pattern-monitor alert examples (raw_dialog/row_dialog) when available
        try:
            monitor_rows = self.monitor.examples(req.pattern_tag, {"date_from": req.date_from, "date_to": req.date_to, "top_n": 5}).get("rows", [])
        except Exception:
            monitor_rows = []
        if not monitor_rows:
            return []
        mdf = pd.DataFrame(monitor_rows)
        if "row_dialog" in mdf.columns and "raw_dialog" not in mdf.columns:
            mdf["raw_dialog"] = mdf["row_dialog"]
        return _from_df(mdf)

    def _build_summary(
        self,
        total: int,
        expected: int,
        delta_abs: int,
        delta_pct: float | None,
        contribution: list[dict],
        risk: PatternRisk,
        area: PrimaryArea | None,
        examples: list[AlertExampleCard],
    ) -> SummaryBlock:
        sign = "выше" if delta_abs >= 0 else "ниже"
        pct_text = "н/д" if delta_pct is None else f"{abs(delta_pct) * 100:.1f}%"
        top_names = ", ".join([r["category"] for r in contribution[:3]]) if contribution else "без выраженного лидера"
        bullets = [
            f"За выбранный период получено {total} жалоб, это на {pct_text} {sign} baseline ({delta_abs:+d}).",
            f"Основной вклад в изменение дали категории: {top_names}.",
            f"Состояние pattern risk: {risk.display_label} ({risk.label}).",
            self._pattern_risk_phrase(risk.label),
        ]
        if area is not None:
            bullets.append(f"Для первичной проверки рекомендуется зона/контур: {area.label}.")
        actions = [
            "Проверить категорию с максимальным вкладом в рост и связанные процессы.",
            "Провести точечный разбор 3–5 примеров жалоб для подтверждения причины роста.",
        ]
        if not examples:
            actions.append("Включить examples для просмотра реальных жалоб в отчете.")
        return SummaryBlock(headline="Краткий вывод", bullets=bullets, recommended_actions=actions)

    def _definitions(self) -> dict[str, str]:
        return {
            "baseline": "Базовый уровень для сравнения: сколько жалоб обычно ожидается в аналогичном периоде.",
            "expected": "Ожидаемое количество жалоб с учетом выбранного режима сравнения.",
            "delta": "Разница между фактическим и ожидаемым количеством жалоб.",
            "delta_pct": "Процентное отклонение фактического значения от ожидаемого.",
            "anomaly": "Нетипичное отклонение относительно обычной динамики.",
            "pattern_risk": "Признак того, что в жалобах сохраняется нетипичный проблемный сценарий.",
            "category_contribution": "Насколько каждая категория повлияла на общий рост жалоб.",
            "categories_above_baseline": "Категории, где жалоб заметно больше ожидаемого уровня.",
            "special_pattern": "Повторяющийся нетипичный подтип жалоб, требующий внимания.",
            "owner": "Наиболее вероятная зона или команда для первичной проверки.",
            "priority": "Приоритет разбора категории или примера на основе вклада в рост.",
            "alert_example": "Реальная жалоба, иллюстрирующая ключевой риск или рост.",
        }

    def build_executive_markdown(
        self,
        summary: SummaryBlock,
        total: int,
        delta_abs: int,
        delta_pct: float | None,
        contribution: list[dict],
        risk: PatternRisk,
        area: PrimaryArea | None,
    ) -> str:
        lines = ["# Executive report", "", f"Всего жалоб: **{total}**", f"Отклонение к baseline: **{delta_abs:+d}** ({'н/д' if delta_pct is None else f'{delta_pct * 100:.1f}%'})", f"Pattern risk: **{risk.label}**"]
        if area:
            lines.append(f"Основной контур: **{area.label}**")
        lines.extend(["", "## Краткий вывод", f"**{summary.headline}**"])
        lines.extend([f"- {b}" for b in summary.bullets])
        lines.extend(["", "## Рекомендованные действия"])
        lines.extend([f"- {a}" for a in summary.recommended_actions])
        lines.extend(["", "## Вклад категорий"])
        lines.extend([f"- {r['category']}: {r['delta']:+.0f}" for r in contribution[:8]])
        return "\n".join(lines)

    def build_executive_html(
        self,
        summary: SummaryBlock,
        total: int,
        delta_abs: int,
        delta_pct: float | None,
        contribution: list[dict],
        risk: PatternRisk,
        area: PrimaryArea | None,
    ) -> str:
        bullets = "".join(f"<li>{b}</li>" for b in summary.bullets)
        actions = "".join(f"<li>{a}</li>" for a in summary.recommended_actions)
        contrib = "".join(f"<li><b>{r['category']}</b>: {r['delta']:+.0f}</li>" for r in contribution[:8])
        area_block = f"<p><b>Основной контур:</b> {area.label}</p>" if area else ""
        return (
            "<html><body style='font-family:Inter,Arial,sans-serif;padding:24px;'>"
            "<h1>Executive report</h1>"
            f"<p><b>Всего жалоб:</b> {total}</p>"
            f"<p><b>Отклонение к baseline:</b> {delta_abs:+d} ({'н/д' if delta_pct is None else f'{delta_pct * 100:.1f}%'})</p>"
            f"<p><b>Pattern risk:</b> {risk.label}</p>"
            f"{area_block}"
            f"<h2>{summary.headline}</h2><ul>{bullets}</ul>"
            f"<h3>Рекомендованные действия</h3><ul>{actions}</ul>"
            f"<h3>Вклад категорий</h3><ul>{contrib}</ul>"
            "</body></html>"
        )

    @staticmethod
    def _pattern_risk_phrase(label: str) -> str:
        if label == "high":
            return "Сохраняются сильные признаки продолжающегося нетипичного проблемного сценария."
        if label == "medium":
            return "Есть отдельные признаки сохранения нетипичного проблемного сценария, требуется проверка."
        if label == "low":
            return "Признаки продолжающегося нетипичного проблемного сценария выражены слабо."
        return "Оценка pattern risk недоступна: недостаточно данных pattern monitoring."

    def _to_markdown(self, report_type: str, sections: list[dict]) -> str:
        lines = [f"# {report_type.title()} report"]
        for s in sections:
            lines.append(f"\n## {s['title']}\n")
            lines.append(str(s["body"]))
        return "\n".join(lines)

    def _to_html(self, report_type: str, sections: list[dict]) -> str:
        body = "".join(f"<section><h2>{s['title']}</h2><pre>{s['body']}</pre></section>" for s in sections)
        return f"<html><body><h1>{report_type.title()} report</h1>{body}</body></html>"
