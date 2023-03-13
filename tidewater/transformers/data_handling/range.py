from typing import Any, Optional, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tidewater.datatypes import Scores, Labels, TimeSeries
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class Range(Transformer):
    """
    The Range Transformer chooses a predefined subset of the input TimeSeries.

    Attributes
    ----------
    start_index : int
        Start index for range
    end_index : int
        End index for range
    """

    def __init__(self, start_index: int, end_index: int):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries = self.get_input_value("data")[0]
        assert timeseries is not None and isinstance(
            timeseries, TimeSeries
        ), "The input values for the SlidingWindow are not valid."

        self.set_output_value(data=TimeSeries(ndarray=timeseries.ndarray[self.start_index : self.end_index]))

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=TimeSeries)
