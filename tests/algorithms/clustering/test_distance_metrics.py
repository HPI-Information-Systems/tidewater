import unittest

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from tidewater.datatypes import TimeSeries
from tidewater.transformers.clusterings.kmedoids import KMedoids
from tidewater.transformers.clusterings.distance_metrics import DistanceMetric, Interpolation


class TestDistanceMetrics(unittest.TestCase):
    def test_matrix(self):
        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(10) + 5 for _ in range(5)] + [np.random.rand(11) - 5 for _ in range(5)],
            )
        )

        should_be = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                dist = DistanceMetric.DTW(data[i].ndarray, data[j].ndarray)
                should_be[i, j] = dist

        generated = DistanceMetric.DTW.matrix([d.ndarray for d in data])
        npt.assert_array_equal(should_be, generated)

    def test_lengths_validation_raises(self):
        data = [np.array([1, 2, 3]), np.array([1, 2])]

        with self.assertRaises(AssertionError):
            DistanceMetric.EUCLIDEAN.matrix(data)

    def test_lengths_validation_not_raises(self):
        data = [np.array([1, 2, 3]), np.array([1, 2, 3])]
        self.assertIsNotNone(DistanceMetric.EUCLIDEAN.matrix(data))

    def test_interpolation_up(self):
        data = [np.sin(np.arange(10)), np.cos(np.arange(11))]

        arr = Interpolation.UP(data)
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(data[0][-1], arr[0, -1])

    def test_interpolation_down(self):
        data = [np.sin(np.arange(10)), np.cos(np.arange(11))]

        arr = Interpolation.DOWN(data)
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(data[1][-1], arr[1, -1])
