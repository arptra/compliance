from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path):
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)
    data.setdefault("training", {})["model_dir"] = str(tmp_path / "models")
    pd.DataFrame({"event_time": ["2025-01-01"], "category": ["A"]}).to_parquet(tmp_path / "all_prepared.parquet", index=False)
    d = tmp_path / "pattern_monitor_latest"
    d.mkdir()
    pd.DataFrame({
        'date': ['2025-01-01','2025-01-01','2025-01-02','2025-01-02'],
        'category': ['A','A','B','B'],
        'subcategory': ['S1','S2','S1','S2'],
        'pattern_like_score': [0.91,0.2,0.85,0.1],
        'event_similarity':[0.8,0.2,0.7,0.1],
        'normal_distance':[0.6,0.1,0.5,0.1],
        'is_pattern_alert':[True,True,True,True],
        'row_dialog':['a','b','c','d']
    }).to_parquet(d / 'scored_rows.parquet', index=False)
    pd.DataFrame({'date':['2025-01-01'],'category':['A'],'pressure':[0.7]}).to_parquet(d/'category_daily_pressure.parquet', index=False)
    pd.DataFrame({'date':['2025-01-01'],'state':[0.5]}).to_parquet(d/'overall_daily_state.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return cfg


def test_train_and_activate(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    rows = client.get('/api/pattern-monitor/alerts', params={'pattern_tag':'latest'}).json()['rows']
    labels = ['true','false','true','false']
    for row, v in zip(rows, labels):
        client.post('/api/feedback', json={'row_id': row['row_id'], 'pattern_tag':'latest', 'verdict': v})
    tr = client.post('/api/pattern-monitor/calibrator/train', json={'pattern_tag':'latest','activate_if_better': True})
    assert tr.status_code == 200
    versions = client.get('/api/pattern-monitor/calibrator/versions').json()
    assert len(versions) >= 1
    active = [v for v in versions if v['active'] == 1]
    assert active
