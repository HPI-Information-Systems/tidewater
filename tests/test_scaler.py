import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import Scores, Labels
from tidewater.datatypes.base import NumpyType
from tidewater.transformers.data_handling.scaler import MinMaxScaler, OrgMinMaxScaler


class TestScaler(unittest.TestCase):
    def setUp(self) -> None:
        self.array = np.array([5, 5, 7.1, 8, 0.0])
        self.expected = OrgMinMaxScaler().fit_transform(self.array.reshape(-1, 1))

    def test_scaling_1d(self):
        f = tempfile.TemporaryDirectory()

        tr = MinMaxScaler.Training()
        tr.set_results_dir(Path(f.name))
        tr.set_input_value(data=NumpyType(ndarray=self.array))
        tr.execute()

        sc = MinMaxScaler()
        sc.set_input_value(model=tr.get_output_value("model")[0])
        sc.set_input_value(data=NumpyType(ndarray=self.array))
        sc.execute()

        array: Optional[NumpyType] = sc.get_output_value("data")[0]

        npt.assert_equal(array.ndarray, self.expected)
