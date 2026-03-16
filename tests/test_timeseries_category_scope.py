from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path):
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    data['prepare']['output_parquet'] = str(tmp_path / 'all_prepared.parquet')
    data.setdefault('analysis', {}).setdefault('pattern_monitoring', {})['interim_dir'] = str(tmp_path)
    rows = []
    for n in range(12):
        cat = chr(ord('A') + n)
        for i in range(12 - n):
            rows.append({'event_time': f'2025-01-{(i%10)+1:02d}', 'category': cat})
    pd.DataFrame(rows).to_parquet(tmp_path / 'all_prepared.parquet', index=False)
    cfg = tmp_path / 'project.yaml'
    cfg.write_text(yaml.safe_dump(data), encoding='utf-8')
    return cfg


def test_category_scope_modes(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    top = client.get('/api/timeseries/by-category').json()
    assert len(top['resolved_categories']) == 10
    custom = client.get('/api/timeseries/by-category', params=[('category_mode', 'custom'), ('categories', 'B')]).json()
    assert set(r['category'] for r in custom['rows']) <= {'B'}
    all_rows = client.get('/api/timeseries/by-category', params={'category_mode': 'all'}).json()['rows']
    cats = set(r['category'] for r in all_rows)
    assert len(cats) >= 12
