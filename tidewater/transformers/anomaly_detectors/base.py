from __future__ import annotations
from typing import Any

from ..base import Transformer
from ..interface import InputInterface, OutputInterface
from ...datatypes import TimeSeries, Scores, Labels
from ...datatypes.base import NumpyType
from ...datatypes.model import Model


class AnomalyDetector(Transformer):
    """
    A base Transformer for all anomaly detection algorithms.
    """

    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert data is not None and isinstance(data, NumpyType), "Input data is not a filled NumpyType"
        scores = self._transform(data.ndarray, **kwargs)
        self.set_output_value(data=Scores(ndarray=scores))

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries, model=Model)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Scores)

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries, labels=Labels)

    @staticmethod
    def _build_train_output_interface() -> OutputInterface:
        return OutputInterface(model=Model)


class UnsupervisedAnomalyDetector(AnomalyDetector):
    """
    A base Transformer for all unsupervised anomaly detection algorithms.
    """

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)


class SemiSupervisedAnomalyDetector(AnomalyDetector):
    """
    A base Transformer for all semi-supervised anomaly detection algorithms.
    """

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)


class SupervisedAnomalyDetector(AnomalyDetector):
    """
    A base Transformer for all supervised anomaly detection algorithms.
    """

    pass
