import unittest

import numpy as np
import numpy.testing as npt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from tidewater.datatypes import TimeSeries
from tidewater.transformers.clusterings.kmeans import KMeans


class TestClusteringKMeans(unittest.TestCase):
    def test_execution(self):
        data = list(
            map(
                lambda x: TimeSeries(ndarray=x),
                [np.random.rand(10) + 5 for _ in range(50)] + [np.random.rand(11) - 5 for _ in range(50)],
            )
        )

        m = KMeans(random_state=42)
        m.set_input_value(data=data)
        m.execute()
        tw_result = m.get_output_value("data")[0].ndarray

        o = TimeSeriesKMeans(random_state=42, metric="dtw")
        data = to_time_series_dataset([x.ndarray.reshape(-1) for x in data])
        o_result = o.fit_predict(data)

        npt.assert_equal(tw_result, o_result)
