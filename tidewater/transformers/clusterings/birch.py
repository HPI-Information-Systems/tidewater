from typing import Any, Optional, List

import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import Birch as SLBirch
from pydantic.dataclasses import dataclass
from tslearn.utils import to_time_series_dataset

from .distance_metrics import DistanceMetric, Interpolation
from .base import DynSizeClustering


@dataclass
class Birch(DynSizeClustering):
    n_clusters: int = 3
    threshold: float = 0.5
    branching_factor: int = 50
    random_state: Optional[int] = None
    n_jobs: int = 1
    interpolation: Optional[Interpolation] = None

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            X = self.interpolation(X)  # type: ignore

        model = SLBirch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor,
        )
        y: np.ndarray = model.fit_predict(X)

        return y
