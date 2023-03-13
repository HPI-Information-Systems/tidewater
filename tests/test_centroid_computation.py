import unittest

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.clusterings.centroid_computation.mean import Mean


class TestCentroidComputation(unittest.TestCase):
    def test_mean(self):
        timeseries = [
            TimeSeries(ndarray=np.array([0, 0.5, 0])),
            TimeSeries(ndarray=np.array([0, 1.0, 0])),
            TimeSeries(ndarray=np.array([0, 1.5, 0])),
            TimeSeries(ndarray=np.array([0, 0.5, 3.5])),
            TimeSeries(ndarray=np.array([0, 1.0, 2.5])),
            TimeSeries(ndarray=np.array([0, 1.5, 3.0])),
        ]

        expected = [np.array([0, 1.0, 0]), np.array([0, 1.0, 3.0])]

        labels = Labels(ndarray=np.array([0, 0, 0, 1, 1, 1]))

        mean = Mean()
        mean.set_input_value(data=timeseries, labels=labels)
        mean.execute()
        centroids = mean.get_output_value("data")[0]

        for c, e in zip(centroids, expected):
            npt.assert_equal(c.ndarray, e)
