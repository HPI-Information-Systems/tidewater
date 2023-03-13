from typing import Any, Optional, List

import numpy as np
from tslearn.clustering import KShape as OrgKShape
from pydantic.dataclasses import dataclass

from .distance_metrics import Interpolation
from .base import DynSizeClustering


@dataclass
class KShape(DynSizeClustering):
    n_clusters: int = 3
    max_iter: int = 100
    random_state: Optional[int] = None
    interpolation: Optional[Interpolation] = None

    def __post_init__(self) -> None:
        super().__init__()

    def _validate_input(self, X: List[np.ndarray]) -> None:
        try:
            np.concatenate(X)
        except:
            raise ValueError("KShape cannot handle variable length subsequences.")

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            X = self.interpolation(X)  # type: ignore
        model = OrgKShape(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=self.random_state)
        labels: np.ndarray = model.fit_predict(X)
        return labels
