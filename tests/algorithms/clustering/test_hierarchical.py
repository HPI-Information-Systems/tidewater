import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.utils import gen_batches

from tidewater.transformers.clusterings.hierarchical import Hierarchical


def test_bandwidth():
    data = np.random.rand(100, 10)
    true_bandwidth = estimate_bandwidth(data)

    distance_matrix = [np.linalg.norm(data[i] - data[j]) for i in range(len(data) - 1) for j in range(i + 1, len(data))]
    pred_bandwidth = Hierarchical._estimate_bandwidth(distance_matrix, len(data))

    assert true_bandwidth == pred_bandwidth
