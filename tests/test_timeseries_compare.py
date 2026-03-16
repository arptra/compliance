from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path):
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path)
    pd.DataFrame(
        {
            'event_time': pd.date_range('2025-01-01', periods=20, freq='D').tolist() + pd.date_range('2025-01-01', periods=20, freq='D').tolist(),
            'category': ['A'] * 20 + ['B'] * 20,
        }
    ).to_parquet(tmp_path / 'all_prepared.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data), encoding='utf-8')
    return cfg


def test_compare_summary_and_contrib(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    r = client.get('/api/timeseries/compare', params={'date_from': '2025-01-11', 'date_to': '2025-01-20'})
    assert r.status_code == 200
    body = r.json()
    assert 'summary' in body
    assert 'contributions' in body
