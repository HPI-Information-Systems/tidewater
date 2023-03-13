from typing import Any, Optional, List, Dict
import matplotlib.pyplot as plt

import numpy as np
import numpy.testing as npt
from tslearn.clustering import TimeSeriesKMeans
from pydantic.dataclasses import dataclass
from dataclasses import field
from tslearn.utils import to_time_series_dataset

from ..interface import OutputInterface
from ...datatypes import Labels, TimeSeries
from .distance_metrics import DistanceMetric, Interpolation
from .base import DynSizeClustering


@dataclass
class KMedoids(DynSizeClustering):
    n_clusters: int = 2
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    max_iter: int = 50
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    def _sample_initial_centroids(self, distance_matrix: np.ndarray) -> np.ndarray:
        centers = []

        # Sample the first point
        initial_index = np.random.choice(range(distance_matrix.shape[0]))
        centers.append(initial_index)

        # Loop and select the remaining points
        for i in range(self.n_clusters - 1):
            distance = distance_matrix[centers]
            dist_min = np.min(distance, axis=0)
            centroid_new = np.argmax(dist_min, axis=0)
            centers.append(centroid_new)

        return np.array(centers)

    def _calculate_medoid(self, assignments: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
        cluster_centers = []
        for c in range(self.n_clusters):
            assigned_ids = np.where(assignments == c)[0]
            medoid = np.argmin(distance_matrix[assigned_ids][:, assigned_ids].sum(axis=0))
            cluster_centers.append(assigned_ids[medoid])
        return np.array(cluster_centers)

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        distance_matrix = self.metric.matrix(
            X, n_jobs=self.n_jobs, verbose=self.verbose, interpolation=self.interpolation, **self.algorithm_args
        )

        cluster_center_ids = self._sample_initial_centroids(distance_matrix)

        for _ in range(self.max_iter):
            assignments: np.ndarray = np.argmin(distance_matrix[cluster_center_ids], axis=0)
            moved_cluster_center_ids = self._calculate_medoid(assignments, distance_matrix)

            if not np.abs(cluster_center_ids - moved_cluster_center_ids).sum() > 0:
                break

            cluster_center_ids = moved_cluster_center_ids

        centroids = [TimeSeries(ndarray=X[cid]) for cid in cluster_center_ids]
        self.set_output_value(centroids=centroids)

        return assignments

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=Labels, centroids=List[TimeSeries])  # type: ignore
