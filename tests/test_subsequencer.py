import unittest
from typing import Optional, List, Tuple

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.subsequencer import Subsequencer


class TestSubsequencer(unittest.TestCase):
    def test_standard(self):
        ts = TimeSeries(ndarray=np.sin(np.linspace(0, 8 * np.pi, 1000)).reshape(-1, 1))
        labels = Labels(ndarray=np.zeros(1000))
        subsequencer = Subsequencer(min_window_size=200)
        subsequencer.set_input_value(data=ts, labels=labels)
        subsequencer.execute()

        cut_ts, cut_labels = subsequencer.get_output_value("data", "labels")
        self.assertEqual(len(cut_ts), 4)
        self.assertEqual(len(cut_labels), 4)
