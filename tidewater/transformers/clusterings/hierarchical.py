from typing import Any, Callable, Dict, Optional, List, Type
from joblib import Parallel, delayed

import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.utils import gen_batches
from tslearn.clustering import TimeSeriesKMeans
from pydantic.dataclasses import dataclass
from dataclasses import field
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from tslearn.metrics import dtw, soft_dtw
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .base import DynSizeClustering
from ...datatypes import TimeSeries, Labels
from ..interface import InputInterface, InterfaceValue
from .distance_metrics import DistanceMetric, Interpolation

import matplotlib.pyplot as plt


@dataclass
class Hierarchical(DynSizeClustering):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    method: str = "single"  # options: single, complete, average, weighted, centroid, median, ward
    n_clusters: Optional[int] = None
    bandwidth: Optional[float] = None
    random_state: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    @staticmethod
    def _estimate_bandwidth(distance_matrix: np.ndarray, len_X: int, quantile: float = 0.3) -> float:
        full_distance_matrix = np.zeros((len_X, len_X))
        last_end = 0
        for i in range(len_X - 1):
            values = distance_matrix[last_end : last_end + len_X - (i + 1)]
            full_distance_matrix[i, i + 1 :] = values
            full_distance_matrix[i + 1 :, i] = values
            last_end = last_end + len_X - (i + 1)

        n_neighbors = int(len_X * quantile)
        bandwidth = 0.0
        for batch_ids in gen_batches(len_X, 500):
            batch = full_distance_matrix[batch_ids]
            partition = np.argpartition(batch, n_neighbors, axis=1)
            nearest = np.array([batch[i, partition[i, :n_neighbors]] for i in range(len(partition))])
            bandwidth += nearest.max(axis=1).sum()  # for each point in batch max distance

        return bandwidth / len_X

    def _calculate_linkings(self, X: List[np.ndarray]) -> np.ndarray:
        distance_matrix = self.metric.z_matrix(
            X, verbose=self.verbose, n_jobs=self.n_jobs, interpolation=self.interpolation, **self.algorithm_args
        )
        linkings: np.ndarray = linkage(np.array(distance_matrix), method=self.method)
        return linkings

    def _cut_tree(self, linkings: np.ndarray, X: List[np.ndarray]) -> np.ndarray:
        if self.n_clusters is not None:
            assignments: np.ndarray = cut_tree(linkings, n_clusters=self.n_clusters).reshape(-1)
        elif self.bandwidth is not None:
            assignments = cut_tree(linkings, height=self.bandwidth).reshape(-1)
        else:
            if self.bandwidth is None:
                if self.interpolation is not None:
                    X = self.interpolation(X)  # type: ignore
                self.bandwidth = estimate_bandwidth(X)
            assignments = cut_tree(linkings, height=self.bandwidth)
        return assignments

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        linkings = self._calculate_linkings(X)
        assignments = self._cut_tree(linkings, X)

        return assignments


class HierarchicalPlotting(Hierarchical):
    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        linkings = self._calculate_linkings(X)
        self._plot(linkings)
        assignments = self._cut_tree(linkings, X)

        return assignments

    def _plot(self, linkings: np.ndarray) -> None:
        labels: Optional[InterfaceValue] = self.get_input_value("labels")[0]
        assert labels is not None and isinstance(
            labels, Labels
        ), f"{self.__class__.__name__} has not received all its inputs yet."

        link_cols: Dict[int, str] = {}
        for i, i12 in enumerate(linkings[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linkings) else ("red" if labels.ndarray[x] == 1 else "blue") for x in i12)
            link_cols[i + 1 + len(linkings)] = c1 if c1 == c2 else "grey"

        dendrogram(linkings, color_threshold=None, link_color_func=lambda k: link_cols[k])
        plt.show()

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels)  # type: ignore
