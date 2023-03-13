import unittest
from pathlib import Path
from typing import Optional
import os

import numpy as np
import pandas as pd
import numpy.testing as npt

from tidewater.datatypes import TimeSeries
from tidewater.transformers.data_handling.loader import CSVTimeSeriesLoader, HDF5TimeSeriesLoader
from tidewater.transformers.data_handling.range import Range


class TestRange(unittest.TestCase):
    def test_range(self):
        ts = TimeSeries(ndarray=np.array([1, 2, 3]))
        range = Range(1, 3)
        range.set_input_value(data=ts)
        range.execute()

        cut_ts: Optional[TimeSeries] = range.get_output_value("data")[0]
        npt.assert_equal(cut_ts.ndarray, np.array([2, 3]))
