from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_template(template_name: str, out_path: str, context: dict) -> None:
    tpl_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(tpl_dir)), autoescape=select_autoescape(["html", "xml"]))
    tpl = env.get_template(template_name)
    html = tpl.render(**context)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")


def write_md(path: str, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
