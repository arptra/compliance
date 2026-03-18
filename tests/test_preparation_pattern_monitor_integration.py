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

    d = tmp_path / 'interim' / 'pattern_monitor_latest'
    d.mkdir(parents=True)
    pd.DataFrame({'date': ['2025-01-01'], 'category': ['A'], 'is_alert': [True], 'raw_dialog': ['CLIENT: text']}).to_parquet(d / 'scored_rows.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-01'], 'category': ['A'], 'pressure': [1.0]}).to_parquet(d / 'category_daily_pressure.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-01'], 'smoothed_state': [0.5], 'overall_pressure': [0.4]}).to_parquet(d / 'overall_daily_state.parquet', index=False)

    src = pd.DataFrame({
        'created_at': ['2025-01-01 10:00:00'],
        'dialog_text': ['CLIENT: проблема 1'],
        'call_text': [''],
        'comment_text': [''],
        'summary_text': [''],
        'subject': ['s1'],
        'channel': ['chat'],
        'product': ['loan'],
        'status': ['new'],
    })
    src.to_excel(tmp_path / 'upload.xlsx', index=False)
    return cfg


def test_preparation_pattern_monitor_block_and_allow(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))

    with open(tmp_path / 'upload.xlsx', 'rb') as f:
        up = client.post('/api/preparation/upload', files={'file': ('upload.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    upload_id = up.json()['upload_id']

    blocked = client.post(f'/api/preparation/jobs/{upload_id}/open-pattern-monitor')
    assert blocked.status_code == 200
    assert blocked.json()['allowed'] is False

    summary_blocked = client.get('/api/pattern-monitor/summary', params={'upload_id': upload_id, 'pattern_tag': 'latest'})
    assert summary_blocked.status_code == 200
    assert summary_blocked.json()['allowed'] is False

    run = client.post(f'/api/preparation/{upload_id}/run')
    assert run.status_code == 200
    assert run.json()['status'] == 'succeeded'

    preset = client.post(f'/api/preparation/jobs/{upload_id}/open-pattern-monitor').json()
    assert preset['allowed'] is True
    assert preset['pattern_monitor_preset']['upload_id'] == upload_id

    summary_ready = client.get('/api/pattern-monitor/summary', params={'upload_id': upload_id, 'pattern_tag': 'latest'})
    assert summary_ready.status_code == 200
    assert summary_ready.json()['allowed'] is True
