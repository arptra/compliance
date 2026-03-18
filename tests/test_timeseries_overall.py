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
            'event_time': pd.date_range('2025-01-01', periods=15, freq='D'),
            'category': ['A'] * 8 + ['B'] * 7,
        }
    ).to_parquet(tmp_path / 'all_prepared.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data), encoding='utf-8')
    return cfg


def test_timeseries_overall_expected_delta(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    r = client.get('/api/timeseries/overall', params={'date_from': '2025-01-08', 'date_to': '2025-01-15', 'baseline_mode': 'previous_period'})
    assert r.status_code == 200
    body = r.json()
    assert len(body['actual']) > 0
    assert len(body['delta']) > 0
    assert any((r.get('value', 0) or 0) > 0 for r in body['expected'])
