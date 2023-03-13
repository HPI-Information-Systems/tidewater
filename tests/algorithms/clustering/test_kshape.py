import unittest

import numpy as np
import numpy.testing as npt

from tidewater.datatypes import TimeSeries
from tidewater.transformers.clusterings.kshape import KShape, OrgKShape


class TestClusteringKShape(unittest.TestCase):
    def test_execution(self):
        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(10) + 5 for _ in range(50)] + [np.random.rand(10) - 5 for _ in range(50)],
            )
        )

        m = KShape(random_state=42)
        m.set_input_value(data=data)
        m.execute()
        tw_result = m.get_output_value("data")[0].ndarray

        o = OrgKShape(random_state=42)
        data = [x.ndarray.reshape(-1) for x in data]
        o_result = o.fit_predict(data)

        npt.assert_equal(tw_result, o_result)

    def test_validation(self):
        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(10) + 5 for _ in range(50)] + [np.random.rand(11) - 5 for _ in range(50)],
            )
        )

        m = KShape(random_state=42)
        m.set_input_value(data=data)
        with self.assertRaises(ValueError):
            m.execute()
