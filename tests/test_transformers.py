import unittest
from pathlib import Path
from typing import Type, Any, Optional

import numpy as np
from pydantic import ValidationError

from tidewater.datatypes import TimeSeries, Scores
from tidewater.datatypes.base import NumpyType, BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer, TransformerMode
from tidewater.transformers.interface import InputInterface


class DummyTransformer(Transformer):
    def execute(self, **kwargs: Any) -> None:
        pass

    @property
    def _numpy_return_type(self) -> Type[NumpyType]:
        return Scores

    @property
    def _numpy_input_type(self) -> Type[NumpyType]:
        return TimeSeries

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)


class TestTransformers(unittest.TestCase):
    def test_transformer_interface_is_ready(self):
        ddt = DummyTransformer()
        ddt.set_input_value(data=TimeSeries(ndarray=np.ndarray([1, 2, 3])))
        self.assertTrue(ddt.is_ready_for_execution)

    def test_transformer_interface_value_validation_fails(self):
        ddt = DummyTransformer()
        with self.assertRaises(ValidationError):
            ddt.set_input_value(data=Model(path=Path(".")))

    def test_transformer_modes(self):
        ddt = DummyTransformer.Training()
        self.assertEqual(ddt._transformer_mode, TransformerMode.TRAINING)
        ddt = DummyTransformer.Transforming()
        self.assertEqual(ddt._transformer_mode, TransformerMode.TRANSFORMING)
