from typing import Any, Optional, List, Type, Union

import numpy as np

from sklearn.preprocessing import StandardScaler
from tidewater.datatypes import Scores, Labels, TimeSeries
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface, InterfaceValue


class Normalizer(Transformer):
    """
    The Normalizer Transformer standard scales the input Timeseries.

    Attributes
    ----------
    """

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries: Optional[InterfaceValue] = self.get_input_value("data")[0]
        assert timeseries is not None and (
            isinstance(timeseries, TimeSeries) or isinstance(timeseries, list)
        ), "The input values for the Normalizer are not valid."

        if type(timeseries) == TimeSeries:
            self.set_output_value(data=TimeSeries(ndarray=StandardScaler().fit_transform(timeseries.ndarray)))
        elif type(timeseries) == list:
            self.set_output_value(
                data=[TimeSeries(ndarray=StandardScaler().fit_transform(ts.ndarray)) for ts in timeseries]
            )

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=Union[TimeSeries, List[TimeSeries]])  # type: ignore

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Union[TimeSeries, List[TimeSeries]])  # type: ignore
