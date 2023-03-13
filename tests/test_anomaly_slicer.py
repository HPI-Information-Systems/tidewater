import unittest
from typing import Optional, List

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.anomaly_slicer import AnomalySlicer


class TestAnomalySlicer(unittest.TestCase):
    def test_slicing(self):
        slicer = AnomalySlicer()

        ts = TimeSeries(ndarray=np.array([0, 0, 0, 1, 2, 3, 0, 0, 4, 5]))
        l = Labels(ndarray=np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]))
        expected = [TimeSeries(ndarray=np.array([1, 2, 3])), TimeSeries(ndarray=np.array([4, 5]))]

        slicer.set_input_value(timeseries=ts, labels=l)
        slicer.execute()
        anomalies: Optional[List[TimeSeries]] = slicer.get_output_value("data")[0]
        assert anomalies is not None, "Anomalies not sliced out of TimeSeries"

        self.assertEqual(len(anomalies), len(expected))
        for act, exp in zip(anomalies, expected):
            npt.assert_equal(act.ndarray, exp.ndarray)

    def test_min_len(self):
        slicer = AnomalySlicer(min_len=2)

        ts = TimeSeries(ndarray=np.array([0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 4, 5]))
        l = Labels(ndarray=np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]))
        expected = [TimeSeries(ndarray=np.array([1, 2, 3])), TimeSeries(ndarray=np.array([4, 5]))]

        slicer.set_input_value(timeseries=ts, labels=l)
        slicer.execute()
        anomalies: Optional[List[TimeSeries]] = slicer.get_output_value("data")[0]
        assert anomalies is not None, "Anomalies not sliced out of TimeSeries"

        self.assertEqual(len(anomalies), len(expected))
        for act, exp in zip(anomalies, expected):
            npt.assert_equal(act.ndarray, exp.ndarray)
