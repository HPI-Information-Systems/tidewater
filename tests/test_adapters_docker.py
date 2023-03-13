import unittest
import pytest
from pathlib import Path
from typing import Type, Any, Optional

import numpy as np
import numpy.testing as npt
from pydantic import ValidationError

from tidewater.datatypes import TimeSeries, Scores
from tidewater.datatypes.base import NumpyType, BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.adapters import (
    UnsupervisedDockerAdapter,
    SemiSupervisedDockerAdapter,
    SupervisedDockerAdapter,
)
from tidewater.transformers.interface import InputInterface
from tidewater.transformers.anomaly_detectors import KMeans


class DummyDockerTransformer(UnsupervisedDockerAdapter):
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def __init__(self):
        super().__init__("dummy_image")

    @property
    def _numpy_return_type(self) -> Type[NumpyType]:
        return Scores

    @property
    def _numpy_input_type(self) -> Type[NumpyType]:
        return TimeSeries

    def _train(self, **kwargs):
        pass

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)


class DummyDockerTransformerSS(SemiSupervisedDockerAdapter, DummyDockerTransformer):
    def _train(self, **kwargs):
        pass


class DummyDockerTransformerS(SupervisedDockerAdapter, DummyDockerTransformer):
    def _train(self, **kwargs):
        pass


class TestAdaptersDocker(unittest.TestCase):
    def test_docker_adapters_interface(self):
        ddt = DummyDockerTransformer()
        ddt.set_input_value(data=TimeSeries(ndarray=np.ndarray([1, 2, 3])))
        self.assertTrue(ddt.is_ready_for_execution)

    def test_docker_adapters_interface_value_validation_fails(self):
        ddt = DummyDockerTransformerSS()
        with self.assertRaises(ValidationError):
            ddt.set_input_value(data=Model(path=Path(".")))

    def test_correct_numpy_format(self):
        ddt = DummyDockerTransformerS()

        X = np.array([[1], [2], [3]])
        expected = np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0]])
        npt.assert_equal(ddt._to_correct_numpy_format(X), expected)

        X = np.array([[1], [2], [3]])
        y = np.array([[0], [0], [1]])
        expected = np.array([[0, 1, 0], [1, 2, 0], [2, 3, 1]])
        npt.assert_equal(ddt._to_correct_numpy_format(X, y), expected)

    @pytest.mark.docker
    def test_result_to_output(self):
        m = KMeans()
        x = TimeSeries(ndarray=np.random.rand(1000))
        m.set_input_value(data=x)
        m.execute()
        scores: Optional[Scores] = m.get_output_value("data")[0]
        self.assertTrue(scores.ndarray.shape[0] > 0)
