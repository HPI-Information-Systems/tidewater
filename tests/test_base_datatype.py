import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from tidewater.datatypes.base import NumpyType


class TestBaseDataType(unittest.TestCase):
    def test_file_handling(self):
        data = np.array([1, 2, 3])
        a = NumpyType(ndarray=data)

        with tempfile.NamedTemporaryFile() as fp:
            a.to_file(Path(fp.name))
            b = NumpyType.from_file(Path(fp.name))

        npt.assert_equal(b.ndarray, data)

    def test_to_2d_from_1d(self):
        data = NumpyType(ndarray=np.array([1, 2, 3]))
        expected = np.array([[1], [2], [3]])

        npt.assert_equal(data.to_2d(), expected)

    def test_to_2d_from_2d(self):
        data = NumpyType(ndarray=np.array([[1], [2], [3]]))
        expected = np.array([[1], [2], [3]])

        npt.assert_equal(data.to_2d(), expected)

    def test_to_2d_from_3d(self):
        data = NumpyType(ndarray=np.array([[[1], [2], [3]]]))

        with self.assertRaises(ValueError):
            data.to_2d()
