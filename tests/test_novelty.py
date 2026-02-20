import numpy as np

from app.novelty import novelty_threshold_from_baseline


def test_percentile_threshold():
    sims = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    th = novelty_threshold_from_baseline(sims, 20)
    assert 0.1 <= th <= 0.25
