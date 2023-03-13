from typing import Any, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass
from meanshift_rs import MeanShift as MeanShiftRS

from .distance_metrics import Interpolation
from .base import DynSizeClustering


@dataclass
class MeanShift(DynSizeClustering):
    n_threads: int = 1
    distance_measure: str = "dtw"
    interpolation: Optional[Interpolation] = None

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            X = self.interpolation(X)  # type: ignore

        model = MeanShiftRS(n_threads=self.n_threads, distance_measure=self.distance_measure)
        model.fit(X)

        return np.array(model.labels)
