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
    pd.DataFrame({'date':['2025-01-01','2025-01-02'],'category':['A','B'],'subcategory':['S1','S2'],'pattern_like_score':[0.9,0.2],'event_similarity':[0.8,0.2],'normal_distance':[0.5,0.1],'is_pattern_alert':[True,True],'row_dialog':['a','b']}).to_parquet(d/'scored_rows.parquet', index=False)
    pd.DataFrame({'date':['2025-01-01'],'category':['A'],'pressure':[0.7]}).to_parquet(d/'category_daily_pressure.parquet', index=False)
    pd.DataFrame({'date':['2025-01-01'],'state':[0.5]}).to_parquet(d/'overall_daily_state.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return cfg


def test_reranked_mode_fallback_then_available(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    r0 = client.get('/api/pattern-monitor/alerts', params={'pattern_tag':'latest','scoring_mode':'reranked'})
    assert r0.status_code == 200
    assert r0.json()['scoring_mode_effective'] == 'base'

    rows = client.get('/api/pattern-monitor/alerts', params={'pattern_tag':'latest'}).json()['rows']
    client.post('/api/feedback/bulk', json={'rows':[{'row_id':rows[0]['row_id'],'pattern_tag':'latest','verdict':'true'},{'row_id':rows[1]['row_id'],'pattern_tag':'latest','verdict':'false'}]})
    client.post('/api/pattern-monitor/calibrator/train', json={'pattern_tag':'latest','activate_if_better':True})
    r1 = client.get('/api/pattern-monitor/alerts', params={'pattern_tag':'latest','scoring_mode':'reranked'})
    assert r1.status_code == 200
    assert r1.json()['scoring_mode_effective'] == 'reranked'
