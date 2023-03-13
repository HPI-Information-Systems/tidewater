import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from tidewater.datatypes.base import NumpyType
from tidewater.transformers.data_handling.writer import Writer


class TestWriter(unittest.TestCase):
    def test_writing_to_file(self):
        n = NumpyType(ndarray=np.array([1, 2, 3, 4]))

        with tempfile.NamedTemporaryFile() as f:
            w = Writer(Path(f.name))
            w.set_input_value(data=n)
            w.execute()

            loaded = np.loadtxt(f.name)

        npt.assert_equal(loaded, n.ndarray)
