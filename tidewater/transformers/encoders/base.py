from abc import abstractmethod
from typing import List, Any, Union, Optional
import numpy as np

from ...datatypes.model import Model
from ...datatypes.base import NumpyType
from ..interface import InputInterface, OutputInterface
from ..base import Transformer


class Encoder(Transformer):
    """
    This is a base class for Encoder Transformers. They encode data into a different space representation.
    """

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        raise NotImplementedError("This method is not implemented for Encoder.")

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError("This method is not implemented for Encoder.")

    @abstractmethod
    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        ...

    def execute(self, **kwargs: Any) -> None:
        input_data = self.get_input_value("data")[0]
        assert (
            input_data is not None and isinstance(input_data, List) and isinstance(input_data[0], NumpyType)
        ), f"{self.__class__.__name__} has not received all its inputs yet."

        input_type = input_data[0].__class__

        data: List[np.ndarray] = [nt.flatten() for nt in input_data]
        encoded = self._encode(data, **kwargs)
        self.set_output_value(data=[input_type(ndarray=arr) for arr in encoded])

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[NumpyType])

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[NumpyType])
