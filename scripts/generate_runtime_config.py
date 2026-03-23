from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepare-output", required=True)
    ap.add_argument("--interim-dir", required=True)
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--exports-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    cfg.setdefault("prepare", {})["output_parquet"] = args.prepare_output
    cfg.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = args.interim_dir
    cfg["analysis"]["pattern_monitoring"]["reports_dir"] = args.reports_dir
    cfg["analysis"]["pattern_monitoring"]["exports_dir"] = args.exports_dir
    cfg.setdefault("training", {})["model_dir"] = args.models_dir

    Path(args.output).write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"runtime config written to {args.output}")


if __name__ == "__main__":
    main()
