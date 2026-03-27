from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def test_meta_prepare_preview_endpoint(tmp_path: Path):
    data = yaml.safe_load(Path('configs/project.yaml').read_text(encoding='utf-8'))
    parquet_path = tmp_path / 'all_prepared.parquet'
    data['prepare']['output_parquet'] = str(parquet_path)

    df = pd.DataFrame(
        {
            'row_id': ['r1', 'r2', 'r3'],
            'client_first_message': ['ошибка входа', 'оплата не проходит', 'вопрос по тарифу'],
            'category': ['login', 'payment', 'other'],
            'event_time': ['2025-01-01 10:00:00', '2025-01-02 10:00:00', '2025-01-03 10:00:00'],
        }
    )
    df.to_parquet(parquet_path, index=False)

    cfg_path = tmp_path / 'project.yaml'
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')

    client = TestClient(create_app(str(cfg_path)))

    resp = client.get('/api/meta/prepare-preview', params={'page': 1, 'page_size': 2, 'q': 'оплата'})
    assert resp.status_code == 200
    body = resp.json()

    assert body['total'] == 1
    assert body['page'] == 1
    assert body['page_size'] == 2
    assert 'row_id' in body['columns']
    assert body['items'][0]['row_id'] == 'r2'
