from typing import Any, Dict, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
from logging import warn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from tidewater.datatypes.base import NumpyType
from tidewater.datatypes.labels import Labels

from tidewater.transformers.interface import InputInterface

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation


@dataclass
class PlotPCASpace(Encoder):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})
    plot: bool = True

    def __post_init__(self) -> None:
        super().__init__()

    def _generate_landmarks(self, data: List[np.ndarray]) -> List[np.ndarray]:
        return data

    def _median(self, data: List[np.ndarray]) -> np.ndarray:
        min_len = min([len(d) for d in data])
        matrix = np.array([d[:min_len] for d in data])
        median = np.median(matrix, axis=0)
        return self.metric.matrix_other(data, [median])

    def _get_extreme_landmarks(self, data: List[np.ndarray]) -> List[np.ndarray]:
        min_len = min([len(d) for d in data])
        cut_data = [d[:min_len] for d in data]
        median = np.median(cut_data, axis=0)
        distances = np.abs(cut_data - median).sum(axis=1)
        idx = np.argsort(distances)
        return idx[:10]

    def _get_labels(self) -> np.ndarray:
        labels = self.get_input_value("labels")[0]
        return labels.ndarray

    def _plot_pca_space(self, data: List[np.ndarray], matrix: np.ndarray) -> None:
        """Plot the PCA space of the data.

        Parameters
        ----------
        data : List[np.ndarray]
            The data to plot.
        matrix : np.ndarray
            The distance matrix of the data.

        Returns
        -------
        None

        See Also
        --------
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

        """

        pca = PCA(n_components=2)
        pca.fit(matrix)
        dim_red = pca.transform(matrix)

        # plt.scatter(dim_red[:, 0], dim_red[:, 1], c=np.abs(pca.components_[0]), cmap="plasma")
        plt.scatter(dim_red[:, 0], dim_red[:, 1], c=self._get_labels())
        # median = pca.transform(self._median(data).reshape(1, -1))
        # plt.scatter(median[:, 0], median[:, 1], c="red")
        # landmarks = pca.transform(matrix[self._get_extreme_landmarks(data)])
        # plt.scatter(landmarks[:, 0], landmarks[:, 1], c="green")
        plt.show()
        print("Most important dimensions:", np.argsort(np.abs(pca.components_[0]))[::-1][:50])

    def _distance_matrix_lazy(self, data: List[np.ndarray], grid: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        """Calculate the distance matrix between the data and the grid if an already saved version is not already calculated."""
        file_name = Path("./distance_matrix.npy")
        if file_name.exists():
            distance_matrix = np.load(file_name)
            return distance_matrix

        n_jobs = kwargs.get("n_jobs", self.n_jobs)
        verbose = kwargs.get("verbose", self.verbose)
        distance_matrix = self.metric.matrix_other(data, grid, verbose=verbose, n_jobs=n_jobs, **self.algorithm_args)
        np.save(file_name, distance_matrix)

        return distance_matrix

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            warn("Interpolation has no effect in the GridSpace(Encoder).")

        grid = self._generate_landmarks(data)
        distance_matrix = self._distance_matrix_lazy(data, grid, **kwargs)
        if self.plot:
            self._plot_pca_space(data, distance_matrix)
        return distance_matrix

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[NumpyType], labels=Labels)
