import unittest
from typing import Optional, List, Tuple

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.subsequence_merger import SubsequenceMerger


class TestSubsequenceMerger(unittest.TestCase):
    def test_standard(self):
        ts = [
            TimeSeries(ndarray=np.array([0, 0, 0])),
            TimeSeries(ndarray=np.array([0, 0, 1])),
            TimeSeries(ndarray=np.array([0, 1, 0])),
        ]
        other = [
            TimeSeries(ndarray=np.array([0, 1, 1])),
            TimeSeries(ndarray=np.array([1, 0, 0])),
            TimeSeries(ndarray=np.array([1, 0, 1])),
        ]
        merger = SubsequenceMerger()
        merger.set_input_value(data=ts, other=other)
        merger.execute()

        data = merger.get_output_value("data")[0]
        npt.assert_array_equal(
            [d.ndarray for d in data],
            [
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([0, 1, 0]),
                np.array([0, 1, 1]),
                np.array([1, 0, 0]),
                np.array([1, 0, 1]),
            ],
        )
