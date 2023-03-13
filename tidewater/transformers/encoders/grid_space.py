from typing import Any, Dict, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
from logging import warn

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation


@dataclass
class GridSpace(Encoder):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    mode: str = "min"  # possible options: ["min", "max"]
    samples: int = 50
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    def _generate_grid(self, data: List[np.ndarray]) -> np.ndarray:
        if self.mode == "min":
            min_len = min(len(x) for x in data)
            combined = np.array([x[:min_len] for x in data])
        elif self.mode == "max":
            max_len = max(len(x) for x in data)
            combined = np.array([np.pad(x, (0, max_len - len(x)), "constant", constant_values=np.nan) for x in data])

        start = np.nanmin(combined, axis=0)
        stop = np.nanmax(combined, axis=0)

        return np.linspace(start, stop, self.samples)

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            warn("Interpolation has no effect in the GridSpace(Encoder).")

        grid = self._generate_grid(data)
        distance_matrix = self.metric.matrix_other(
            data, grid, verbose=self.verbose, n_jobs=self.n_jobs, **self.algorithm_args
        )
        return distance_matrix
