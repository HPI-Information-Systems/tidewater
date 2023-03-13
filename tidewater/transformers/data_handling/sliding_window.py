from typing import Any, Optional, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tidewater.datatypes import Scores, Labels, TimeSeries
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class SlidingWindow(Transformer):
    """
    The SlidingWindow Transformer generates a List of windows from the input TimeSeries.

    Attributes
    ----------
    window_size : int
        The window size for the moving windows
    stride : int
        The stride for moving windows
    """

    def __init__(self, window_size: int, stride: int = 1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries = self.get_input_value("data")[0]
        assert timeseries is not None and isinstance(
            timeseries, TimeSeries
        ), "The input values for the SlidingWindow are not valid."

        slid_windows = sliding_window_view(timeseries.ndarray, self.window_size, axis=0)
        windows = [TimeSeries(ndarray=x.reshape(self.window_size, -1)) for x in slid_windows[:: self.stride]]
        self.set_output_value(data=windows)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries])
