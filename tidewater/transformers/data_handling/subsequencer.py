from typing import Any, Optional, List, Sequence

import numpy as np
from scipy.signal import find_peaks

from tidewater.datatypes import TimeSeries, Labels
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class Subsequencer(Transformer):
    """
    The Subsequencer Transformer generates a List of windows from the input TimeSeries
    based on automatic cycle detection.

    Attributes
    ----------
    min_window_size : int
        The minimum window size for the cycles
    channel : int
        The index of the channel to search for cycles (default: 0)
    """

    def __init__(self, min_window_size: int, channel: int = 0):
        super().__init__()
        self.min_window_size = min_window_size
        self.channel = channel

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries = self.get_input_value("data")[0]
        assert timeseries is not None and isinstance(
            timeseries, TimeSeries
        ), "The input values for the SlidingWindow are not valid."

        labels = self.get_input_value("labels")[0]
        assert labels is not None and isinstance(
            labels, Labels
        ), "The input values for the SlidingWindow are not valid."

        peaks = find_peaks(timeseries.ndarray[:, self.channel], distance=self.min_window_size)[0]
        windows = np.split(timeseries.ndarray, peaks)
        if labels is not None:
            label_windows = np.split(labels.ndarray, peaks)
        middle_peak = int(np.median(peaks[1:] - peaks[:-1]))
        if len(windows[0]) < (middle_peak / 2):
            windows = windows[1:]
            if labels is not None:
                label_windows = label_windows[1:]

        if len(windows[0]) > (middle_peak * 1.3):
            windows[0] = windows[0][:-middle_peak]
            if labels is not None:
                label_windows[0] = label_windows[0][:-middle_peak]

        if len(windows[-1]) < (middle_peak / 2):
            windows = windows[:-1]
            if labels is not None:
                label_windows = label_windows[:-1]

        if len(windows[-1]) > (middle_peak * 1.3):
            windows[-1] = windows[-1][:middle_peak]
            if labels is not None:
                label_windows[-1] = label_windows[-1][:middle_peak]

        windows = [TimeSeries(ndarray=x) for x in windows]
        if labels is not None:
            label_windows = [Labels(ndarray=x) for x in label_windows]
            self.set_output_value(labels=label_windows)
        self.set_output_value(data=windows)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries, labels=Labels)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries], labels=List[Labels])  # type: ignore
