from abc import abstractmethod
from typing import Any, Optional, List

import numpy as np

from tidewater.datatypes import TimeSeries, Labels
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class CentroidComputation(Transformer):
    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def _calculate_centroid(self, timeseries: List[np.ndarray], labels: np.ndarray) -> List[np.ndarray]:
        ...

    def execute(self, **kwargs: Any) -> None:
        timeseries: List[TimeSeries]
        labels: Labels
        timeseries, labels = self.get_input_value("data", "labels")  # type: ignore
        assert timeseries is not None and labels is not None, "Input variable are not set for Centroid Computation"
        data = [ts.ndarray for ts in timeseries]
        centroids = self._calculate_centroid(data, labels.ndarray)
        self.set_output_value(data=list(map(lambda x: TimeSeries(ndarray=x), centroids)))

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels)  # type: ignore

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries])
