import unittest

import numpy as np
from sklearn.metrics import rand_score

from tidewater.datatypes import TimeSeries
from tidewater.transformers.clusterings.distance_metrics import DistanceMetric
from tidewater.transformers.clusterings.kmedoids import KMedoids


class TestClusteringKMedoids(unittest.TestCase):
    def test_execution(self):
        labels = ([0] * 50) + ([1] * 50)

        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(2) + 2 for _ in range(50)] + [np.random.rand(2) - 2 for _ in range(50)],
            )
        )

        m = KMedoids(metric=DistanceMetric.EUCLIDEAN, random_state=42, n_clusters=2)
        m.set_input_value(data=data)
        m.execute()
        tw_result = m.get_output_value("data")[0].ndarray

        self.assertGreaterEqual(rand_score(tw_result, labels), 0.98)
