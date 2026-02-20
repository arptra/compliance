from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from .compare import compare_month
from .config import load_config
from .infer_month import infer_month
from .prepare_dataset import prepare_dataset
from .train_models import train
from .trends import build_trends

app = typer.Typer(help="complaints-trends CLI")
console = Console()


@app.command("prepare")
def prepare_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    pilot: bool = typer.Option(False, "--pilot"),
    month: str | None = typer.Option(None, "--month"),
    date_from: str | None = typer.Option(None, "--date-from"),
    date_to: str | None = typer.Option(None, "--date-to"),
    limit: int | None = typer.Option(None, "--limit"),
    mock_llm: bool = typer.Option(False, "--mock-llm"),
):
    cfg = load_config(config)
    df = prepare_dataset(cfg, pilot=pilot, month=month, date_from=date_from, date_to=date_to, limit=limit or cfg.prepare.pilot_limit, llm_mock=mock_llm)
    console.log(f"Prepared rows: {len(df)}")


@app.command("train")
def train_cmd(config: str = typer.Option(..., "--config", help="Path to project yaml config")):
    cfg = load_config(config)
    metrics = train(cfg)
    console.log(metrics)


@app.command("trends")
def trends_cmd(config: str = typer.Option(..., "--config", help="Path to project yaml config")):
    cfg = load_config(config)
    m = build_trends(cfg)
    console.log(m.tail(3))


@app.command("infer-month")
def infer_month_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    excel: str = typer.Option(..., "--excel"),
    month: str = typer.Option(..., "--month"),
):
    cfg = load_config(config)
    df = infer_month(cfg, excel, month)
    console.log(f"Inferred rows: {len(df)}")


@app.command("compare")
def compare_cmd(
    config: str = typer.Option(..., "--config", help="Path to project yaml config"),
    new_month: str = typer.Option(..., "--new-month"),
    baseline_range: str = typer.Option(..., "--baseline-range"),
):
    cfg = load_config(config)
    df = compare_month(cfg, new_month, baseline_range)
    console.log(f"Novel rows: {int(df['is_novel'].sum())}")


@app.command("demo")
def demo_cmd(config: str = typer.Option("configs/project.yaml", "--config", help="Path to project yaml config")):
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

    prepare_dataset(cfg, pilot=True, month="2025-10", limit=200, llm_mock=True)
    prepare_dataset(cfg, pilot=False, llm_mock=True)
    train(cfg)
    build_trends(cfg)
    infer_month(cfg, f"{cfg.input.input_dir}/2025-12.xlsx", "2025-12")
    compare_month(cfg, "2025-12", "2025-10..2025-11")
    console.log("Demo pipeline completed")


if __name__ == "__main__":
    app()
