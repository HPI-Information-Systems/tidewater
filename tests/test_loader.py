import unittest
from pathlib import Path
from typing import Optional
import os

import numpy as np
import pandas as pd
import numpy.testing as npt

from tidewater.datatypes import TimeSeries
from tidewater.transformers.data_handling.loader import CSVTimeSeriesLoader, HDF5TimeSeriesLoader


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(os.getcwd())

    def test_loading_csv(self):
        expected = pd.read_csv(self.base / "tests/data/dataset.csv").values
        l = CSVTimeSeriesLoader(self.base / "tests/data/dataset.csv")
        l.execute()
        loaded: Optional[TimeSeries] = l.get_output_value("data")[0]

        npt.assert_equal(loaded.ndarray, expected)

    def test_loading_input_availability(self):
        l = CSVTimeSeriesLoader(self.base / "tests/data/dataset.csv")
        self.assertTrue(l.is_ready_for_execution)

    def test_loading_with_index(self):
        path = self.base / "tests/data/taxi.csv"
        expected = pd.read_csv(path)
        l = CSVTimeSeriesLoader(path, index_col=True)
        l.execute()

        npt.assert_equal(l.get_output_value("data")[0].ndarray, expected.iloc[:, 1:].values)

    def test_loading_with_labels(self):
        path = self.base / "tests/data/taxi.csv"
        expected = pd.read_csv(path)
        l = CSVTimeSeriesLoader(path, label_col=2)
        l.execute()

        data, labels = l.get_output_value("data", "labels")

        npt.assert_equal(data.ndarray, expected.iloc[:, :-1].values)
        npt.assert_equal(labels.ndarray, expected.iloc[:, 2].values)

    def test_dropping_csv(self):
        path = self.base / "tests/data/taxi.csv"
        l = CSVTimeSeriesLoader(path, drop=["is_anomaly"])
        l.execute()
        data = l.get_output_value("data")[0]

        self.assertTrue(data.ndarray.shape[1] == 2)

    def test_dropping_h5(self):
        l = HDF5TimeSeriesLoader(self.base / "tests/data/test.h5", drop=["a"])
        l.execute()
        loaded: Optional[TimeSeries] = l.get_output_value("data")[0]

        npt.assert_equal(loaded.ndarray, np.array([[1, 2]]))

    def test_loading_h5(self):
        l = HDF5TimeSeriesLoader(self.base / "tests/data/test.h5")
        l.execute()
        loaded: Optional[TimeSeries] = l.get_output_value("data")[0]

        npt.assert_equal(loaded.ndarray, np.array([[0, 1, 2]]))

    def test_loading_with_labels_h5(self):
        l = HDF5TimeSeriesLoader(self.base / "tests/data/test.h5", label_col="c")
        l.execute()
        loaded, labels = l.get_output_value("data", "labels")

        npt.assert_equal(loaded.ndarray, np.array([[0, 1]]))
        npt.assert_equal(labels.ndarray, np.array([2]))

    def test_only_csv(self):
        path = self.base / "tests/data/taxi.csv"
        l = CSVTimeSeriesLoader(path, only=["value"])
        l.execute()
        data = l.get_output_value("data")[0]

        self.assertTrue(data.ndarray.shape[1] == 1)

    def test_only_h5(self):
        l = HDF5TimeSeriesLoader(self.base / "tests/data/test.h5", only=["a"])
        l.execute()
        loaded: Optional[TimeSeries] = l.get_output_value("data")[0]

        npt.assert_equal(loaded.ndarray, np.array([[0]]))
