from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path / 'interim')
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data['llm']['enabled'] = False
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')

    src = pd.DataFrame({
        'created_at': ['2025-01-01 10:00:00', '2025-01-02 12:00:00'],
        'dialog_text': ['CLIENT: проблема 1', 'CLIENT: проблема 2'],
        'call_text': ['', ''],
        'comment_text': ['', ''],
        'summary_text': ['', ''],
        'subject': ['s1', 's2'],
        'channel': ['chat', 'chat'],
        'product': ['loan', 'loan'],
        'status': ['new', 'new'],
    })
    excel_path = tmp_path / 'upload.xlsx'
    src.to_excel(excel_path, index=False)
    return cfg


def test_preparation_upload_run_and_merge(tmp_path: Path):
    cfg = _setup(tmp_path)
    client = TestClient(create_app(str(cfg)))

    with open(tmp_path / 'upload.xlsx', 'rb') as f:
        up = client.post('/api/preparation/upload', files={'file': ('upload.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    assert up.status_code == 200
    upload_id = up.json()['upload_id']

    run = client.post(f'/api/preparation/{upload_id}/run')
    assert run.status_code == 200
    assert run.json()['status'] == 'succeeded'

    jobs = client.get('/api/preparation/jobs').json()['jobs']
    job = next(j for j in jobs if j['upload_id'] == upload_id)
    assert job['available_for_pattern_monitor'] is True
    assert job['prepared_rows'] > 0

    main = pd.read_parquet(tmp_path / 'all_prepared.parquet')
    assert 'source_upload_id' in main.columns
    assert (main['source_upload_id'] == upload_id).any()
