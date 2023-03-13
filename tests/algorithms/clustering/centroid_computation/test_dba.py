import unittest

import numpy as np
import numpy.testing as npt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.barycenters import softdtw_barycenter

from tidewater.datatypes import TimeSeries, Labels
from tidewater.transformers.clusterings.centroid_computation.dba import DBA


class TestDBA(unittest.TestCase):
    def test_execution(self):
        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(10) + 5 for _ in range(50)] + [np.random.rand(11) - 5 for _ in range(50)],
            )
        )
        labels = Labels(ndarray=np.zeros(100))

        m = DBA()
        m.set_input_value(data=data, labels=labels)
        m.execute()
        tw_result = m.get_output_value("data")[0][0].ndarray

        o_result = softdtw_barycenter([d.ndarray for d in data])

        npt.assert_equal(tw_result, o_result)
