import unittest
from typing import Any, List, Union

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.clusterings.distance_metrics import DistanceMetric
from tidewater.transformers.encoders.grid_space import GridSpace
from tidewater.transformers.encoders.random_space import RandomSpace


class TestEncoder(unittest.TestCase):
    def test_grid_generation(self):
        input_data = [
            TimeSeries(ndarray=np.array([1, 2, 3])),
            TimeSeries(ndarray=np.array([4, 5, 6])),
            TimeSeries(ndarray=np.array([7, 8, 9])),
        ]

        expected_format = [
            [0, np.sqrt(3 * 6**2)],
            [np.sqrt(3 * 3**2), np.sqrt(3 * 3**2)],
            [np.sqrt(3 * 6**2), 0],
        ]

        dc = GridSpace(metric=DistanceMetric.EUCLIDEAN, samples=2, mode="min")
        dc.set_input_value(data=input_data)
        dc.execute()
        encoded = dc.get_output_value("data")[0]

        for i in range(3):
            npt.assert_equal(encoded[i].ndarray, expected_format[i])

    def test_random_generation(self):
        data = [np.random.rand(10) for _ in range(10)]
        rg = RandomSpace(samples=10)
        lms = rg._generate_landmarks(data)
        for lm in lms:
            self.assertIn(lm.sum(), list(map(lambda x: x.sum(), data)))
