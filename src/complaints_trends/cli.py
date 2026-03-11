from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import typer
from rich.console import Console

from .compare import compare_month
from .config import load_config
from .infer_month import infer_month
from .novelty_hunt import novelty_hunt
from .prepare_dataset import prepare_dataset
from .pattern_fit import run_pattern_fit
from .pattern_monitor import run_pattern_monitor
from .train_models import train
from .trends import build_trends
from .viz.report import build_visual_report, materialize_predictions
from .viz.state import VizPaths
from .viz.viewer import run_viewer

app = typer.Typer(help="complaints-trends CLI")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@app.command("prepare")
def prepare_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    pilot: bool = typer.Option(False, "--pilot"),
    date_from: str | None = typer.Option(None, "--date-from"),
    date_to: str | None = typer.Option(None, "--date-to"),
    limit: int | None = typer.Option(None, "--limit"),
    mock_llm: bool = typer.Option(False, "--mock-llm"),
):
    logger.info("[stage=prepare] start")
    cfg = load_config(config)
    df = prepare_dataset(cfg, pilot=pilot, date_from=date_from, date_to=date_to, limit=limit or cfg.prepare.pilot_limit, llm_mock=mock_llm)
    logger.info("[stage=prepare] done")
    console.log(f"Prepared rows: {len(df)}")


@app.command("train")
def train_cmd(config: str = typer.Option(..., "--config", help="Path to project yaml config")):
    logger.info("[stage=train] start")
    cfg = load_config(config)
    metrics = train(cfg)
    logger.info("[stage=train] done")
    console.log(metrics)


@app.command("trends")
def trends_cmd(config: str = typer.Option(..., "--config", help="Path to project yaml config")):
    logger.info("[stage=trends] start")
    cfg = load_config(config)
    m = build_trends(cfg)
    logger.info("[stage=trends] done")
    console.log(m.tail(3))


@app.command("infer-month")
def infer_month_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    excel: str = typer.Option(..., "--excel"),
    month: str = typer.Option(..., "--month"),
):
    logger.info("[stage=infer-month] start")
    cfg = load_config(config)
    df = infer_month(cfg, excel, month)
    logger.info("[stage=infer-month] done")
    console.log(f"Inferred rows: {len(df)}")


@app.command("compare")
def compare_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    new_month: str = typer.Option(..., "--new-month"),
    baseline_range: str = typer.Option(..., "--baseline-range"),
):
    logger.info("[stage=compare] start")
    cfg = load_config(config)
    df = compare_month(cfg, new_month, baseline_range)
    logger.info("[stage=compare] done")
    console.log(f"Novel rows: {int(df['is_novel'].sum())}")


@app.command("demo")
def demo_cmd(config: str = typer.Option("configs/project.yaml", "--config", help="Path to project yaml config")):
    logger.info("[stage=demo] start")
    cfg = load_config(config)
    Path(cfg.input.input_dir).mkdir(parents=True, exist_ok=True)

    def mk(month: str, n: int):
        rows = []
        for i in range(n):
            bad = i % 3 == 0
            txt = (
                "CLIENT: У меня не работает оплата, ошибка 500, верните деньги.\nOPERATOR: Проверим"
                if bad
                else "CLIENT: Подскажите график работы отделения.\nOPERATOR: 9-18"
            )
            ts = "2025-10-09 12:55:29" if month=="2025-10" else ("2025-11-09 12:55:29" if month=="2025-11" else "2025-12-09 12:55:29")
            rows.append({"dialog_text": txt, "subject": "demo", "channel": "chat", "product": "app", "status": "closed", "created_at": ts})
        pd.DataFrame(rows).to_excel(Path(cfg.input.input_dir) / f"{month}.xlsx", index=False)

    mk("2025-10", 120)
    mk("2025-11", 140)
    mk("2025-12", 80)

    prepare_dataset(cfg, pilot=True, date_from="2025-10-01 00:00:00", date_to="2025-10-31 23:59:59", limit=200, llm_mock=True)
    prepare_dataset(cfg, pilot=False, llm_mock=True)
    train(cfg)
    build_trends(cfg)
    infer_month(cfg, f"{cfg.input.input_dir}/2025-12.xlsx", "2025-12")
    compare_month(cfg, "2025-12", "2025-10..2025-11")
    logger.info("[stage=demo] done")
    console.log("Demo pipeline completed")


@app.command("viz-build")
def viz_build_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    tag: str = typer.Option("latest", "--tag"),
    label_source: str = typer.Option("pred", "--label-source", help="pred|llm"),
    freq: str = typer.Option("D", "--freq", help="D|W|M"),
    top_n: int = typer.Option(12, "--top-n"),
    date_from: str | None = typer.Option(None, "--date-from"),
    date_to: str | None = typer.Option(None, "--date-to"),
    baseline_range: str | None = typer.Option(None, "--baseline-range"),
    new_month: str | None = typer.Option(None, "--new-month"),
    force_materialize: bool = typer.Option(False, "--force-materialize"),
):
    logger.info("[stage=viz-build] start")
    cfg = load_config(config)
    if label_source not in {"pred", "llm"}:
        raise typer.BadParameter("--label-source must be pred or llm")
    if freq not in {"D", "W", "M"}:
        raise typer.BadParameter("--freq must be D, W or M")

    predicted_path = Path("data/interim/all_predicted.parquet")
    if label_source == "pred" and (force_materialize or (not predicted_path.exists())):
        materialize_predictions(cfg, cfg.prepare.output_parquet, predicted_path)

    report_path, state_path = build_visual_report(
        cfg=cfg,
        tag=tag,
        label_source=label_source,
        date_from=date_from,
        date_to=date_to,
        baseline_range=baseline_range,
        new_month=new_month,
        top_n=top_n,
        freq=freq,
    )
    logger.info("[stage=viz-build] done")
    console.log(f"viz report: {report_path}")
    console.log(f"viz state: {state_path}")


@app.command("viz-view")
def viz_view_cmd(
    state: str | None = typer.Option(None, "--state"),
    tag: str = typer.Option("latest", "--tag"),
):
    state_path = Path(state) if state else VizPaths(tag).state_parquet
    if not state_path.exists():
        raise typer.BadParameter(f"state parquet not found: {state_path}")
    run_viewer(state_path, tag=tag)


@app.command("novelty-hunt")
def novelty_hunt_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    new_month: str = typer.Option(..., "--new-month"),
    baseline_range: str = typer.Option(..., "--baseline-range", help="YYYY-MM..YYYY-MM"),
    tag: str = typer.Option("latest", "--tag"),
    use_llm_summary: bool = typer.Option(False, "--use-llm-summary"),
):
    logger.info("[stage=novelty-hunt] start")
    cfg = load_config(config)
    report_path, state_path, export_path = novelty_hunt(
        cfg,
        new_month=new_month,
        baseline_range=baseline_range,
        tag=tag,
        use_llm_summary=use_llm_summary,
    )
    logger.info("[stage=novelty-hunt] done")
    console.log(f"novelty-hunt report: {report_path}")
    console.log(f"novelty-hunt state: {state_path}")
    console.log(f"novelty-hunt export: {export_path}")



@app.command("pattern-fit")
def pattern_fit_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    tag: str = typer.Option(..., "--tag"),
    normal_period: str | None = typer.Option(None, "--normal-period"),
    event_period: str | None = typer.Option(None, "--event-period"),
    label_source: str | None = typer.Option(None, "--label-source", help="llm|pred"),
    use_llm_summary: bool = typer.Option(False, "--use-llm-summary"),
):
    logger.info("[stage=pattern-fit] start")
    cfg = load_config(config)
    pm = cfg.analysis.pattern_monitoring
    src = label_source or pm.label_source
    if src not in {"llm", "pred"}:
        raise typer.BadParameter("--label-source must be llm or pred")
    norm = normal_period or pm.normal_period
    ev = event_period or pm.event_period
    if not norm or not ev:
        raise typer.BadParameter("normal/event period must be set via CLI or config")
    report, fit_bundle, growth = run_pattern_fit(cfg, tag=tag, normal_period=norm, event_period=ev, label_source=src)
    logger.info("[stage=pattern-fit] done")
    console.log(f"pattern-fit report: {report}")
    console.log(f"pattern-fit bundle: {fit_bundle}")
    console.log(f"pattern-fit growth summary: {growth}")


@app.command("pattern-monitor")
def pattern_monitor_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    tag: str = typer.Option(..., "--tag"),
    label_source: str | None = typer.Option(None, "--label-source", help="llm|pred"),
    date_from: str | None = typer.Option(None, "--date-from"),
    date_to: str | None = typer.Option(None, "--date-to"),
    month: str | None = typer.Option(None, "--month"),
    force_materialize: bool = typer.Option(False, "--force-materialize"),
    use_llm_summary: bool = typer.Option(False, "--use-llm-summary"),
):
    logger.info("[stage=pattern-monitor] start")
    cfg = load_config(config)
    pm = cfg.analysis.pattern_monitoring
    src = label_source or pm.label_source
    if src not in {"llm", "pred"}:
        raise typer.BadParameter("--label-source must be llm or pred")
    if month and (date_from or date_to):
        raise typer.BadParameter("Use either --month or --date-from/--date-to")
    scored, state, report = run_pattern_monitor(
        cfg,
        tag=tag,
        label_source=src,
        date_from=date_from,
        date_to=date_to,
        month=month,
        force_materialize=force_materialize,
    )
    logger.info("[stage=pattern-monitor] done")
    console.log(f"pattern-monitor scored rows: {scored}")
    console.log(f"pattern-monitor daily state: {state}")
    console.log(f"pattern-monitor report: {report}")


if __name__ == "__main__":
    app()
