from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _base_cfg(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path)
    pd.DataFrame({'event_time': ['2025-01-01'], 'category': ['A']}).to_parquet(tmp_path / 'all_prepared.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return cfg


def test_pattern_monitor_summary_risk_full(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    d = tmp_path / 'pattern_monitor_latest'
    d.mkdir()
    pd.DataFrame({'date': ['2025-01-01', '2025-01-02'], 'category': ['A', 'A'], 'is_alert': [True, False]}).to_parquet(d / 'scored_rows.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-01'], 'category': ['A'], 'pressure': [1.0]}).to_parquet(d / 'category_daily_pressure.parquet', index=False)
    pd.DataFrame({'date': ['2025-01-02'], 'smoothed_state': [0.5], 'overall_pressure': [0.4]}).to_parquet(d / 'overall_daily_state.parquet', index=False)

    client = TestClient(create_app(str(cfg)))
    r = client.get('/api/pattern-monitor/summary', params={'pattern_tag': 'latest', 'date_from': '2025-01-01', 'date_to': '2025-01-02'})
    assert r.status_code == 200
    s = r.json()['summary']
    assert s['pattern_risk_score'] is not None
    assert s['pattern_risk_calc_mode'] == 'full'
    assert s['pattern_risk_label'] == 'medium'


def test_pattern_monitor_summary_risk_state_only(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    d = tmp_path / 'pattern_monitor_latest'
    d.mkdir()
    pd.DataFrame({'date': ['2025-01-02'], 'overall_pressure': [0.7]}).to_parquet(d / 'overall_daily_state.parquet', index=False)

    client = TestClient(create_app(str(cfg)))
    r = client.get('/api/pattern-monitor/summary', params={'pattern_tag': 'latest', 'date_from': '2025-01-01', 'date_to': '2025-01-02'})
    assert r.status_code == 200
    s = r.json()['summary']
    assert s['pattern_risk_calc_mode'] == 'state_only'
    assert s['pattern_risk_score'] == 0.7
    assert s['pattern_risk_label'] == 'high'


def test_pattern_monitor_summary_risk_unavailable(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    client = TestClient(create_app(str(cfg)))
    r = client.get('/api/pattern-monitor/summary', params={'pattern_tag': 'latest', 'date_from': '2025-01-01', 'date_to': '2025-01-02'})
    assert r.status_code == 200
    s = r.json()['summary']
    assert s['pattern_risk_score'] is None
    assert s['pattern_risk_label'] == 'unavailable'
    assert s['pattern_risk_calc_mode'] == 'unavailable'
