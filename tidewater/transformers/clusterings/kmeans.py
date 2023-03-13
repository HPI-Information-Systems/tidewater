from typing import Any, Optional, List

import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans as SLKMeans
from pydantic.dataclasses import dataclass
from tslearn.utils import to_time_series_dataset

from .distance_metrics import DistanceMetric
from .base import DynSizeClustering


@dataclass
class KMeans(DynSizeClustering):
    n_clusters: int = 3
    max_iter: int = 50
    random_state: Optional[int] = None
    metric: str = "dtw"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        model = TimeSeriesKMeans(
            metric=self.metric,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        labels: np.ndarray = model.fit_predict(to_time_series_dataset(X))
        return labels
