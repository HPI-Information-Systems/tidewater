from typing import Any, Optional

import numpy as np

from tidewater.datatypes import Scores, Labels
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class Thresholder(Transformer):
    """
    The Thresholder generates Labels from Scores by comparing the Scores to a preset threshold. Every value above the
    threshold will be marked '1' everything else is marked '0'.

    Attributes
    ----------
    threshold : float
        The threshold used for comparing
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold: float = threshold

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def execute(self, **kwargs: Any) -> None:
        scores = self.get_input_value("data")[0]
        assert scores is not None and isinstance(scores, Scores), "The input values for the Thresholder are not valid."

        labels = (scores.ndarray > self.threshold).astype(int)
        self.set_output_value(data=Labels(ndarray=labels))

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=Scores)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Labels)
