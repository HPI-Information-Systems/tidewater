from typing import Any, Dict, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation


@dataclass
class DistanceSpace(Encoder):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            data = self.interpolation(data)  # type: ignore

        if self.interpolation is None and Path("distance_matrix.npy").exists():
            distance_matrix = np.load("distance_matrix.npy")
        else:
            distance_matrix = self.metric.matrix(data, verbose=self.verbose, n_jobs=self.n_jobs, **self.algorithm_args)
        return distance_matrix
