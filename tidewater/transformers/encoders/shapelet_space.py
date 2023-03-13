from typing import Any, Dict, Optional, List
from itertools import repeat
from functools import reduce

import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
import stumpy as st

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation

import matplotlib.pyplot as plt


@dataclass
class ShapeletSpace(Encoder):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})
    print: bool = False

    def __post_init__(self) -> None:
        super().__init__()

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            data = self.interpolation(data)  # type: ignore

        init: List[np.ndarray] = []
        cat = np.concatenate(reduce(lambda x, y: x + list(y), zip(data, repeat(np.array([np.nan]))), init))

        m: np.ndarray = st.stump(cat, min(len(x) for x in data))[:, 0].astype(np.float_)
        m_chunked = [m[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(m))]
        mins = np.array([c.min() for c in m_chunked])
        maxs = np.array([c.max() for c in m_chunked])

        assert len(mins) == len(data), "The mins should have the same number of elements as the input data."

        motifs = np.percentile(mins, 5)
        discords = np.percentile(maxs, 99)

        shapelet_ids = np.where((mins <= motifs) | (maxs >= discords))[0]

        if self.verbose:
            print(
                f"Found {len(shapelet_ids)} shapelets ({sum(mins <= motifs)} motifs, {sum(maxs >= discords)} discords)."
            )

        if self.print:
            time = 0
            for i, x in enumerate(data):
                c = "red" if i in shapelet_ids else "blue"
                plt.plot(np.arange(time, time + len(x)), x, color=c)
                time += len(x)
            plt.show()

        distance_matrix = self.metric.matrix_other(
            series=data,
            other=[data[i] for i in shapelet_ids],
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.algorithm_args,
        )

        return distance_matrix
