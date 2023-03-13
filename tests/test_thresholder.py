import unittest
from typing import Optional

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import Scores, Labels
from tidewater.transformers.data_handling.thresholder import Thresholder


class TestThresholder(unittest.TestCase):
    def test_thresholding(self):
        scores = Scores(ndarray=np.array([0.5, 0.5, 0.71, 0.8, 0.0]))
        expected = np.array([0, 0, 1, 1, 0])

        th = Thresholder(threshold=0.7)
        th.set_input_value(data=scores)
        th.execute()

        labels: Optional[Labels] = th.get_output_value("data")[0]
        assert labels is not None and isinstance(labels, Labels), "Output value is not valid"

        npt.assert_equal(labels.ndarray, expected)
