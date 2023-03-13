import unittest
from typing import Optional, List, Tuple

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.normalizer import Normalizer


class TestNormalizer(unittest.TestCase):
    def test_timeseries(self):
        ts = TimeSeries(ndarray=np.sin(np.linspace(0, 8 * np.pi, 1000)).reshape(-1, 1) * 100)
        normalizer = Normalizer()
        normalizer.set_input_value(data=ts)
        normalizer.execute()

        data = normalizer.get_output_value("data")[0]
        self.assertLessEqual(data.ndarray.max(), 2)
        self.assertGreater(data.ndarray.min(), -2)

    def test_timeseries_list(self):
        ts = [TimeSeries(ndarray=np.sin(np.linspace(0, 8 * np.pi, 1000)).reshape(-1, 1) * 100)]
        normalizer = Normalizer()
        normalizer.set_input_value(data=ts)
        normalizer.execute()

        data = normalizer.get_output_value("data")[0][0]
        self.assertLessEqual(data.ndarray.max(), 2)
        self.assertGreater(data.ndarray.min(), -2)
