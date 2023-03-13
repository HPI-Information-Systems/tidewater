from abc import ABC
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import h5py

from ..base import Transformer
from ..interface import OutputInterface, InputInterface
from ...datatypes import TimeSeries, Labels
from ...datatypes.model import Model


class Loader(Transformer, ABC):
    """
    This Transformer loads a pandas dataset from a given path.
    """

    def __init__(
        self,
        path: Path,
        index_col: bool = False,
        label_col: Optional[Union[int, str]] = None,
        drop: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.path = path
        self.index_col = index_col
        self.label_col = label_col
        self.drop = drop or []
        self.only = only or []
        self.kwargs = kwargs

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        """
        This Loader does not have a train step.
        """

        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        This Loader does not have a transform step.
        """

        pass

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface()

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=TimeSeries, labels=Labels)


class CSVTimeSeriesLoader(Loader):
    def __init__(self, path: Path, separator: str = ",", *args: Any, **kwargs: Any) -> None:
        super().__init__(path, *args, **kwargs)
        self.separator = separator

    def _load_dataset_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep=self.separator, **self.kwargs)

    def _extract_labels(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        label_name = self.label_col if type(self.label_col) == str else dataset.columns[self.label_col]
        labels: np.ndarray = dataset.iloc[:, self.label_col].values
        dataset = dataset.drop(label_name, axis=1)
        return dataset, labels

    def _filter_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if len(self.only) > 0:
            dataset = dataset[self.only]
        else:
            for drop_col in self.drop:
                dataset = dataset.drop(drop_col, axis=1)
        return dataset

    def _set_output_data(self, dataset: pd.DataFrame) -> None:
        self.set_output_value(data=TimeSeries(ndarray=dataset.values))

    def _dataset_hook(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset

    def execute(self, **kwargs: Any) -> None:
        dataset = self._load_dataset_dataframe()
        dataset = self._dataset_hook(dataset)
        if self.label_col is not None:
            dataset, labels = self._extract_labels(dataset)
            self.set_output_value(labels=Labels(ndarray=labels))
        dataset = self._filter_columns(dataset)

        if self.index_col:
            dataset = dataset.iloc[:, 1:]

        self._set_output_data(dataset)


class UCRTimeSeriesLoader(CSVTimeSeriesLoader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["separator"] = "\t"
        super().__init__(*args, header=None, label_col=0, **kwargs)

    def _dataset_hook(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # drop rows that have less than 4 non-nan values, 1 for the label and 3 for the time series
        return dataset.dropna(axis=0, thresh=4)

    def _set_output_data(self, dataset: pd.DataFrame) -> None:
        array = dataset.values

        filter_out_nan = lambda x: x[~np.isnan(x)]
        arrays = [filter_out_nan(array[i]) for i in range(array.shape[0])]
        time_series = [TimeSeries(ndarray=a) for a in arrays]

        self.set_output_value(data=time_series)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=List[TimeSeries], labels=Labels)  # type: ignore


class HDF5TimeSeriesLoader(Loader):
    def execute(self, **kwargs: Any) -> None:
        dataset = h5py.File(self.path)
        fields = list(dataset.keys())

        if self.label_col is not None and type(self.label_col) == str:
            labels = np.array(dataset[self.label_col])
            self.set_output_value(labels=Labels(ndarray=labels))
            fields.remove(self.label_col)

        if len(self.only) > 0:
            fields = self.only
        else:
            for drop_col in self.drop:
                fields.remove(drop_col)

        channels = np.stack([np.array(dataset[f]) for f in fields], axis=1)
        self.set_output_value(data=TimeSeries(ndarray=channels))
