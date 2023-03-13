from typing import Any, Optional, List

import numpy as np

from tidewater.datatypes import TimeSeries, Labels
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class AnomalySlicer(Transformer):
    def __init__(self, min_len: int = 2):
        super().__init__()
        self.min_len = min_len

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        ts, l = self.get_input_value("timeseries", "labels")
        assert ts is not None and l is not None, "AnomalySlicer has not received all inputs yet."
        assert isinstance(ts, TimeSeries) and isinstance(l, Labels), "The inputs are of the wrong format."

        tsa: np.ndarray = ts.ndarray
        la: np.ndarray = l.ndarray

        helper_l = np.r_[[0], la.reshape(-1), [0]]
        l_start_idx = np.where(helper_l - np.roll(helper_l, 1) == 1)[0] - 1
        l_end_idx = np.where(helper_l - np.roll(helper_l, -1) == 1)[0] - 1
        assert l_start_idx.shape == l_end_idx.shape, "Start and end idx should have the same shape."

        ranges = zip(l_start_idx, l_end_idx)
        subsequences: List[TimeSeries] = [
            TimeSeries(ndarray=tsa[s : e + 1]) for s, e in ranges if (((e + 1) - s) >= self.min_len)
        ]
        self.set_output_value(data=subsequences)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(timeseries=TimeSeries, labels=Labels)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries])
