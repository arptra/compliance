from __future__ import annotations

from pathlib import Path

import yaml

from complaints_trends.config import load_config
from complaints_trends.api.services.preparation_service import PreparationService


def _cfg(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path / 'interim')
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data['llm']['enabled'] = False
    cfg_path = tmp_path / 'project.yaml'
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return cfg_path


def test_preparation_registry_create_and_list(tmp_path: Path):
    cfg = load_config(str(_cfg(tmp_path)))
    svc = PreparationService(cfg)

    content = b"PK\x03\x04"  # invalid xlsx payload, but upload job should still be created
    r = svc.create_upload_job('new.xlsx', content)
    assert r.upload_id
    jobs = svc.list_preparation_jobs()
    assert any(j.upload_id == r.upload_id for j in jobs)
