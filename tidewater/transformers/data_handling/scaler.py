import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler as OrgMinMaxScaler

from tidewater.datatypes.base import NumpyType
from tidewater.datatypes.model import SKLearnModel
from tidewater.transformers.base import Transformer, TransformerMode
from tidewater.transformers.interface import InputInterface, OutputInterface


class Scaler(Transformer):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.scaler: Optional[Any] = None

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> SKLearnModel:
        self.scaler = self._scaler_class()()
        self.scaler.fit(X)
        fname = str(uuid.uuid1())
        return SKLearnModel.from_transformer(self.scaler, self._intermediate_results_dir / fname)

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self.scaler is None:
            self._train(X)
        if self.scaler is not None:
            result: np.ndarray = self.scaler.transform(X)
            return result
        else:
            raise ValueError("Scaler is not trained")

    def execute(self, **kwargs: Any) -> None:
        array = self.get_input_value("data")[0]
        assert array is not None and isinstance(array, NumpyType), "The input values for the Scaler are not valid."

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            model = self.get_input_value("model")[0]
            assert model is not None and isinstance(
                model, SKLearnModel
            ), "The input values for the Scaler are not valid."
            self.scaler = model.materialize()

            scaled = self._transform(array.to_2d())
            self.set_output_value(data=NumpyType(ndarray=scaled))
        else:
            model = self._train(array.to_2d())
            self.set_output_value(model=model)

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType, model=SKLearnModel)

    @staticmethod
    def _build_train_output_interface() -> OutputInterface:
        return OutputInterface(model=SKLearnModel)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=NumpyType)

    @staticmethod
    @abstractmethod
    def _scaler_class() -> Type:
        """
        This method returns an SKLearn Scaler.
        """
        ...


class MinMaxScaler(Scaler):
    @staticmethod
    def _scaler_class() -> Type:
        return OrgMinMaxScaler  # type: ignore
