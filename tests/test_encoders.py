import unittest
from typing import Any, List, Union

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.encoders.base import Encoder


class DummyEncoder(Encoder):
    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> Union[np.ndarray, List[np.ndarray]]:
        return np.zeros((len(data), 2))


class TestEncoder(unittest.TestCase):
    def test_data_flow(self):
        input_data = [
            TimeSeries(ndarray=np.array([1, 2, 3])),
            TimeSeries(ndarray=np.array([4, 5, 6, 7])),
            TimeSeries(ndarray=np.array([7, 8, 9])),
        ]

        expected_format = [TimeSeries(ndarray=np.zeros(2))] * 3

        dc = DummyEncoder()
        dc.set_input_value(data=input_data)
        dc.execute()
        encoded = dc.get_output_value("data")[0]

        for i in range(3):
            npt.assert_equal(encoded[i].ndarray, expected_format[i].ndarray)
