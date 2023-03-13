import unittest
from pathlib import Path
from typing import Optional
import os

import numpy as np
import pandas as pd
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.data_handling.loader import CSVTimeSeriesLoader, HDF5TimeSeriesLoader
from tidewater.transformers.data_handling.label_reshaper import LabelReshaper


class TestLabelReshaper(unittest.TestCase):
    def test_range(self):
        ts = [
            TimeSeries(ndarray=np.array([0, 0, 0])),
            TimeSeries(ndarray=np.array([0, 2, 2])),
            TimeSeries(ndarray=np.array([1, 1, 0])),
        ]
        reshaper = LabelReshaper()
        reshaper.set_input_value(data=ts)
        reshaper.execute()

        cut_ts: Optional[Labels] = reshaper.get_output_value("data")[0]
        npt.assert_equal(cut_ts.ndarray, np.array([0, 2, 1]))
