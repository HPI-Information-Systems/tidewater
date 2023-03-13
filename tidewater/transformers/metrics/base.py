from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from tidewater.datatypes import Labels
from tidewater.datatypes.base import Scalar
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface, InterfaceValue

from sklearn.metrics import precision_score, recall_score, f1_score, rand_score, homogeneity_score


class Metric(Transformer):
    """
    This Transformer takes labels and calculates the corresponding score.
    """

    def __init__(self, print_info: str = "", **kwargs: Any) -> None:
        super().__init__()
        self.result: Optional[float] = None
        self.print_info = print_info
        self.kwargs = kwargs

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        raise NotImplementedError("This method is not implemented for Metric.")

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError("This method is not implemented for Metric.")

    def execute(self, **kwargs: Any) -> None:
        true_labels: Optional[InterfaceValue] = self.get_input_value("true_labels")[0]
        assert true_labels is not None and isinstance(
            true_labels, Labels
        ), "The input values for the Metric Calculation are not valid."
        pred_labels: Optional[InterfaceValue] = self.get_input_value("pred_labels")[0]
        assert pred_labels is not None and isinstance(
            pred_labels, Labels
        ), "The input values for the Metric Calculation are not valid."

        self.result = self._metric_calculation(true_labels.ndarray, pred_labels.ndarray)
        print(f"{self.print_info}{self.__class__.__name__}: {self.result}")

        self.set_output_value(
            score=Scalar(data=self.result, meta={"metric": self.__class__.__name__, **self.kwargs.get("meta", {})})
        )

    @abstractmethod
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        This method calculates the quality of predictions. It needs to be defined for Metric subclasses.
        """
        ...

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(true_labels=Labels, pred_labels=Labels)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(score=Scalar)


class Precision(Metric):
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        return float(precision_score(true_labels, pred_labels))


class Recall(Metric):
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        return float(recall_score(true_labels, pred_labels))


class F1(Metric):
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        return float(f1_score(true_labels, pred_labels))


class RandScore(Metric):
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        return float(rand_score(true_labels, pred_labels))


class HomogeneityScore(Metric):
    def _metric_calculation(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        return float(homogeneity_score(true_labels, pred_labels))
