from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path)
    pd.DataFrame({
        'event_time': ['2025-01-01', '2025-01-02', '2024-12-30'],
        'category': ['A', 'A', 'A'],
        'text': ['a', 'b', 'c'],
    }).to_parquet(tmp_path / 'all_prepared.parquet', index=False)
    d = tmp_path / 'pattern_monitor_latest'
    d.mkdir()
    pd.DataFrame({'date': ['2025-01-01', '2025-01-02'], 'category': ['A', 'A'], 'is_alert': [True, False], 'raw_dialog': ['raw 1 long message', 'raw 2']}).to_parquet(d / 'scored_rows.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-01'], 'category': ['A'], 'pressure': [1.0]}).to_parquet(d / 'category_daily_pressure.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-02'], 'smoothed_state': [0.9], 'overall_pressure': [0.8]}).to_parquet(d / 'overall_daily_state.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return cfg


def test_report_service_includes_pattern_risk(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))

    # legacy report response
    ops = client.post('/api/reports/operations', json={'filters': {'date_from': '2025-01-01', 'date_to': '2025-01-02', 'pattern_tag': 'latest'}, 'output_format': 'json'})
    assert ops.status_code == 200
    payload = ops.json()
    assert payload['metrics']['pattern_risk_score'] is not None
    assert payload['metrics']['pattern_risk_label'] in {'low', 'medium', 'high'}
    assert any(s['title'] == 'Pattern risk' for s in payload['sections'])

    # executive payload
    ex = client.post('/api/reports/executive', json={'date_from': '2025-01-01', 'date_to': '2025-01-02', 'pattern_tag': 'latest'})
    assert ex.status_code == 200
    ex_payload = ex.json()
    assert ex_payload['kpis']['pattern_risk_score'] is not None
    assert ex_payload['kpis']['pattern_risk_label'] in {'low', 'medium', 'high'}


def test_executive_examples_fallback_to_raw_dialog(tmp_path: Path):
    cfg = _setup(tmp_path)
    # overwrite prepared data without text to force fallback
    pd.DataFrame({
        'event_time': ['2025-01-01', '2025-01-02'],
        'category': ['A', 'A'],
    }).to_parquet(tmp_path / 'all_prepared.parquet', index=False)

    client = TestClient(create_app(str(cfg)))
    ex = client.post('/api/reports/executive', json={'date_from': '2025-01-01', 'date_to': '2025-01-02', 'pattern_tag': 'latest'})
    assert ex.status_code == 200
    rows = ex.json()['charts']['alert_examples']
    assert len(rows) > 0
