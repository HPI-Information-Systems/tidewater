from typing import Any, Optional, List

import numpy as np
from sklearn.cluster import Birch as SKKMeans
from pydantic.dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .base import Clustering


@dataclass
class SameSizeKMeans(Clustering):
    n_clusters: int = 3
    random_state: Optional[int] = None
    n_jobs: int = 1

    def __post_init__(self) -> None:
        super().__init__()

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        size = int(np.ceil(len(X) / self.n_clusters))
        model = SKKMeans(n_clusters=size, threshold=0.1)
        labels = model.fit_predict(X)
        cluster_centers = np.stack([X[labels == i].mean(axis=0) for i in range(size)])
        centers = cluster_centers.reshape(-1, 1, X.shape[-1]).repeat(self.n_clusters, 1).reshape(-1, X.shape[-1])
        distance_matrix = cdist(X, centers)
        centers = linear_sum_assignment(distance_matrix)[1] // self.n_clusters
        return centers
