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


def test_preparation_run_overrides_input_source_to_uploaded_file(tmp_path: Path, monkeypatch):
    cfg = _setup(tmp_path)
    client = TestClient(create_app(str(cfg)))
    captured: dict[str, object] = {}

    def _fake_prepare_dataset(cfg_obj, pilot: bool, llm_mock: bool):
        captured['input_dir'] = cfg_obj.input.input_dir
        captured['file_names'] = cfg_obj.input.file_names
        captured['file_glob'] = cfg_obj.input.file_glob
        captured['file_format'] = cfg_obj.input.file_format
        return pd.DataFrame({
            'event_time': ['2025-01-01 10:00:00'],
            'is_complaint_llm': [True],
            'complaint_category_llm': ['other'],
        })

    monkeypatch.setattr('complaints_trends.api.services.preparation_service.prepare_dataset', _fake_prepare_dataset)

    with open(tmp_path / 'upload.xlsx', 'rb') as f:
        up = client.post('/api/preparation/upload', files={'file': ('upload.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    upload_id = up.json()['upload_id']

    run = client.post(f'/api/preparation/{upload_id}/run')
    assert run.status_code == 200
    assert run.json()['status'] == 'succeeded'

    jobs = client.get('/api/preparation/jobs').json()['jobs']
    job = next(j for j in jobs if j['upload_id'] == upload_id)
    stored_path = Path(job['stored_path'])

    assert captured['input_dir'] == str(stored_path.parent)
    assert captured['file_names'] == [stored_path.name]
    assert captured['file_glob'] == stored_path.name
    assert captured['file_format'] == 'excel'
