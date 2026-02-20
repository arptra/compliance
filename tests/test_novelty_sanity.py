import numpy as np
from scipy.sparse import csr_matrix

from complaints_trends.novelty import compute_novelty_scores


def test_novelty_shapes():
    xb = csr_matrix(np.random.rand(20, 30))
    xn = csr_matrix(np.random.rand(5, 30))
    scores, thr, znew = compute_novelty_scores(xb, xn, "kmeans_distance", 10, 3, 90)
    assert len(scores) == 5
    assert znew.shape[0] == 5
    assert thr >= 0
