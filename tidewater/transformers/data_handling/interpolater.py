from typing import Any, Optional, List, Type, Union

import numpy as np

from sklearn.preprocessing import StandardScaler
from tidewater.datatypes import Scores, Labels, TimeSeries
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.clusterings.distance_metrics import Interpolation
from tidewater.transformers.interface import InputInterface, OutputInterface, InterfaceValue


class Interpolater(Transformer):
    """
    The Interpolater Transformer brings multiple TimeSeries to the same lengths.

    Attributes
    ----------
    interpolation: Interpolation
    """

    def __init__(self, interpolation: Interpolation = Interpolation.DOWN, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.interpolation = interpolation

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries = self.get_input_value("data")[0]
        assert timeseries is not None and (
            isinstance(timeseries, list) and isinstance(timeseries[0], TimeSeries)
        ), "The input values for the Interpolater are not valid."

        arrays = [ts.ndarray.flatten() for ts in timeseries]
        interpolated = self.interpolation(arrays)

        self.set_output_value(data=[TimeSeries(ndarray=ts) for ts in interpolated])

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries])

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries])
