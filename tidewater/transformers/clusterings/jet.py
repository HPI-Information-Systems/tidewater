from typing import Any, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass

from .base import DynSizeClustering
from .distance_metrics import DistanceMetric

from jet import JET as OrgJET


@dataclass
class JET(DynSizeClustering):
    n_clusters: int = 3
    n_pre_clusters: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    metric: DistanceMetric = DistanceMetric.SHAPE_BASED_DISTANCE
    c: float = 700

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        model = OrgJET(
            n_clusters=self.n_clusters,
            n_pre_clusters=self.n_pre_clusters,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            metric=self.metric.name.lower(),
            c=self.c,
        )
        labels: np.ndarray = model.fit_predict(X)
        return labels
