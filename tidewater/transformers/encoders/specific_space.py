from typing import Any, Dict, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
from logging import warn

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation


@dataclass
class SpecificSpace(Encoder):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    indices: List[int] = field(default_factory=lambda: [])
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    def _generate_landmarks(self, data: List[np.ndarray]) -> List[np.ndarray]:
        return [data[i] for i in self.indices]

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            warn("Interpolation has no effect in the GridSpace(Encoder).")

        grid = self._generate_landmarks(data)
        n_jobs = kwargs.get("n_jobs", self.n_jobs)
        verbose = kwargs.get("verbose", self.verbose)
        distance_matrix = self.metric.matrix_other(data, grid, verbose=verbose, n_jobs=n_jobs, **self.algorithm_args)
        return distance_matrix
