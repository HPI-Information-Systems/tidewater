import unittest
from typing import Any

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.clusterings.base import Clustering


class DummyClustering(Clustering):
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.zeros(X.shape[0])


class TestClustering(unittest.TestCase):
    def test_collection_to_matrix(self):
        input_data = [
            TimeSeries(ndarray=np.array([1, 2, 3])),
            TimeSeries(ndarray=np.array([4, 5, 6])),
            TimeSeries(ndarray=np.array([7, 8, 9])),
        ]

        expected_format = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        dc = DummyClustering()
        dc.set_input_value(data=input_data)
        combined_format = dc._collection_to_matrix()

        npt.assert_equal(combined_format, expected_format)

    def test_set_label(self):
        labels = Labels(ndarray=np.array([0, 0, 1]))
        dc = DummyClustering()
        dc._set_predicted_labels(labels)

        npt.assert_equal(dc.get_output_value("data")[0].ndarray, labels.ndarray)

    def test_internal_execute(self):
        data = [TimeSeries(ndarray=np.random.rand(10)) for _ in range(100)]
        dc = DummyClustering()
        dc.set_input_value(data=data)
        dc.execute()

        npt.assert_equal(dc.get_output_value("data")[0].ndarray, np.zeros(100))
