import unittest
from typing import Optional, List, Tuple

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.sliding_window import SlidingWindow


class TestSlidingWindow(unittest.TestCase):
    def test_standard(self):
        ts = TimeSeries(ndarray=np.sin(np.linspace(0, 8 * np.pi, 1000)).reshape(-1, 1))
        subsequencer = SlidingWindow(window_size=200)
        subsequencer.set_input_value(data=ts)
        subsequencer.execute()

        data = subsequencer.get_output_value("data")[0]
        self.assertEqual(len(data), 1000 - 200 + 1)
