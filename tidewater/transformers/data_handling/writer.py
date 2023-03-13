from pathlib import Path
from typing import Any, Optional, List, Union

import numpy as np
import pandas as pd

from tidewater.datatypes.base import NumpyType, Scalar
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface


class Writer(Transformer):
    """
    This Transformer writes a NumpyType to a given path.

    Attributes
    ----------
    path : Path
        The path to the file that is written to
    """

    def __init__(self, path: Path, suffix: str = ""):
        super().__init__()

        self.path = path
        self.suffix = suffix

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert (
            data is not None
            or isinstance(data, NumpyType)
            or (isinstance(data, list) and isinstance(data[0], NumpyType))
        ), "Input data is not filled yet."

        if isinstance(data, NumpyType):
            data.to_file(self.path)
        else:
            for d in data:
                path = Path(str(self.path) + self.suffix)
                d.to_file(path)  # type: ignore

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=Union[NumpyType, List[NumpyType]])  # type: ignore


class ScalarDataFrameAppender(Writer):
    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert data is not None or type(data) == Scalar, "Input data is not filled yet."

        if not self.path.exists():
            df = data.to_dataframe(**kwargs)  # type: ignore
        else:
            df = pd.read_csv(self.path)
            df = pd.concat([df, data.to_dataframe(**kwargs)], ignore_index=True)  # type: ignore
        df.to_csv(self.path, index=False)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=Scalar)


class TimeAndScalarDataFrameAppender(ScalarDataFrameAppender):
    def execute(self, **kwargs: Any) -> None:
        time = self.get_input_value("time")[0]
        assert time is not None or type(time) == Scalar, "Input data is not filled yet."
        kwargs.update({"time": time.data, **time.meta})
        return super().execute(**kwargs)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=Scalar, time=Scalar)
