from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ...config import ProjectConfig
from ...infer_month import infer_month
from ...pattern_fit import run_pattern_fit
from ...pattern_monitor import run_pattern_monitor
from ...viz.report import build_visual_report
from ..schemas import RunRequest, RunResponse


class RunService:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

    def run_viz_build(self, req: RunRequest) -> RunResponse:
        started = datetime.now(timezone.utc)
        p = req.params
        try:
            report, state = build_visual_report(
                cfg=self.cfg,
                tag=p.get("tag", "latest"),
                label_source=p.get("label_source", "pred"),
                date_from=p.get("date_from"),
                date_to=p.get("date_to"),
                baseline_range=p.get("baseline_range"),
                new_month=p.get("new_month"),
                top_n=int(p.get("top_n", 12)),
                freq=p.get("freq", "D"),
            )
            return RunResponse(status="success", started_at=started, finished_at=datetime.now(timezone.utc), outputs={"report": str(report), "state": str(state)}, logs=["viz-build completed"])
        except Exception as e:
            return RunResponse(status="error", started_at=started, finished_at=datetime.now(timezone.utc), outputs={}, logs=[], error=str(e))

    def run_pattern_fit(self, req: RunRequest) -> RunResponse:
        started = datetime.now(timezone.utc)
        p = req.params
        try:
            report, bundle, growth = run_pattern_fit(
                self.cfg,
                tag=p.get("tag", "latest"),
                normal_period=p.get("normal_period") or self.cfg.analysis.pattern_monitoring.normal_period,
                event_period=p.get("event_period") or self.cfg.analysis.pattern_monitoring.event_period,
                label_source=p.get("label_source") or self.cfg.analysis.pattern_monitoring.label_source,
            )
            return RunResponse(status="success", started_at=started, finished_at=datetime.now(timezone.utc), outputs={"report": str(report), "fit_bundle": str(bundle), "growth_summary": str(growth)}, logs=["pattern-fit completed"])
        except Exception as e:
            return RunResponse(status="error", started_at=started, finished_at=datetime.now(timezone.utc), outputs={}, logs=[], error=str(e))

    def run_pattern_monitor(self, req: RunRequest) -> RunResponse:
        started = datetime.now(timezone.utc)
        p = req.params
        try:
            scored, state, report = run_pattern_monitor(
                self.cfg,
                tag=p.get("tag", "latest"),
                label_source=p.get("label_source") or self.cfg.analysis.pattern_monitoring.label_source,
                date_from=p.get("date_from"),
                date_to=p.get("date_to"),
                month=p.get("month"),
                force_materialize=bool(p.get("force_materialize", False)),
            )
            return RunResponse(status="success", started_at=started, finished_at=datetime.now(timezone.utc), outputs={"scored": str(scored), "state": str(state), "report": str(report)}, logs=["pattern-monitor completed"])
        except Exception as e:
            return RunResponse(status="error", started_at=started, finished_at=datetime.now(timezone.utc), outputs={}, logs=[], error=str(e))

    def run_infer_month(self, req: RunRequest) -> RunResponse:
        started = datetime.now(timezone.utc)
        p = req.params
        try:
            df = infer_month(self.cfg, p["excel"], p["month"])
            out_path = Path("data/interim") / f"month_{p['month']}.parquet"
            return RunResponse(status="success", started_at=started, finished_at=datetime.now(timezone.utc), outputs={"rows": str(len(df)), "output": str(out_path)}, logs=["infer-month completed"])
        except Exception as e:
            return RunResponse(status="error", started_at=started, finished_at=datetime.now(timezone.utc), outputs={}, logs=[], error=str(e))
