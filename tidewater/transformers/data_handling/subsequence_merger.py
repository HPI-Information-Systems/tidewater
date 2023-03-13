from typing import Any, Optional, List

import numpy as np

from tidewater.datatypes import TimeSeries
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class SubsequenceMerger(Transformer):
    """
    The SubsequencerMerger Transformer merges two Lists of subsequence windows in one List.

    Attributes
    ----------
    """

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        timeseries = self.get_input_value("data")[0]
        assert timeseries is not None and isinstance(timeseries, list), "The input values for the Merger are not valid."

        other = self.get_input_value("other")[0]
        assert other is not None and isinstance(other, list), "The input values for the Merger are not valid."

        merged = timeseries + other

        self.set_output_value(data=merged)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], other=List[TimeSeries])

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries])
