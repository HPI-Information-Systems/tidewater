from __future__ import annotations
from abc import abstractmethod
from typing import Any, List, Optional

import numpy as np

from ..interface import InputInterface, OutputInterface, InterfaceValue
from ...datatypes import TimeSeries, Labels
from ..base import Transformer
from ...datatypes.model import Model


class Clustering(Transformer):
    """
    A base Transformer for all time series clustering algorithms.
    """

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        """This step is required only for (Semi-)Supervised algorithms!"""
        pass

    def _collection_to_matrix(self) -> np.ndarray:
        """
        Stacks the list of input TimeSeries into one matrix.

        Returns
        -------
        np.ndarray
            Stacked matrix
        """

        input_data: Optional[InterfaceValue] = self.get_input_value("data")[0]
        assert (
            input_data is not None and isinstance(input_data, List) and isinstance(input_data[0], TimeSeries)
        ), f"{self.__class__.__name__} has not received all its inputs yet."
        data = np.vstack([ts.ndarray.flat for ts in input_data])
        return data

    def _collection_to_list(self) -> List[np.ndarray]:
        """
        Stacks the list of input TimeSeries into one matrix.

        Returns
        -------
        np.ndarray
            Stacked matrix
        """

        input_data: Optional[InterfaceValue] = self.get_input_value("data")[0]
        assert (
            input_data is not None and isinstance(input_data, List) and isinstance(input_data[0], TimeSeries)
        ), f"{self.__class__.__name__} has not received all its inputs yet."
        data = [ts.ndarray.reshape(-1) for ts in input_data]
        return data

    def _set_predicted_labels(self, labels: Labels) -> None:
        """
        Sets the predicted Labels as output value.

        Parameters
        ----------
        labels : Labels
            Predicted labels that the algorithm outputs
        """
        self.set_output_value(data=labels)

    def execute(self, **kwargs: Any) -> None:
        data = self._collection_to_matrix()
        labels = self._transform(data, **kwargs)
        self._set_predicted_labels(Labels(ndarray=labels))

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Internal function that is called by the execute function.

        Parameters
        ----------
        X : np.ndarray
            Data to be analyzed

        Returns
        -------
        np.ndarray
            Cluster labels for the data points in _data_
        """

        ...

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries])

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Labels)


class DynSizeClustering(Clustering):
    def execute(self, **kwargs: Any) -> None:
        data = self._collection_to_list()
        labels = self._transform(np.empty(0), data=data, **kwargs)
        self._set_predicted_labels(Labels(ndarray=labels))

    @abstractmethod
    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Internal function that is called by the execute function.

        Parameters
        ----------
        X : List[np.ndarray]
            Data to be analyzed

        Returns
        -------
        np.ndarray
            Cluster labels for the data points in _data_
        """

        ...

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self._cluster_transform(kwargs.get("data", []), **kwargs)
