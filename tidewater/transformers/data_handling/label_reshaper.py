from abc import abstractmethod
from typing import Any, Optional, List

import numpy as np

from tidewater.datatypes import Labels, TimeSeries
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface, InterfaceValue


class LabelReshaper(Transformer):
    """
    This Transformer takes label subsequences and reshapes them to single labels per subsequence.
    """

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        subsequences: Optional[InterfaceValue] = self.get_input_value("data")[0]
        assert subsequences is not None and isinstance(
            subsequences, List
        ), "The input values for the Metric Calculation are not valid."

        labels = np.zeros(len(subsequences))
        for i, subs in enumerate(subsequences):
            classes, counts = np.unique(subs.ndarray, return_counts=True)
            labels[i] = classes[np.argmax(counts)]

        self.set_output_value(data=Labels(ndarray=labels))

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries])

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Labels)
