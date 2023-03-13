from typing import Any, Callable, Dict, Optional, List, Type
from joblib import Parallel, delayed

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.utils import gen_batches
from tslearn.clustering import TimeSeriesKMeans
from pydantic.dataclasses import dataclass
from dataclasses import field
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from tslearn.metrics import dtw, soft_dtw
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .base import DynSizeClustering
from ...datatypes import TimeSeries, Labels
from ..interface import InputInterface, InterfaceValue
from .distance_metrics import DistanceMetric, Interpolation

import matplotlib.pyplot as plt


@dataclass
class DBScan(DynSizeClustering):
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    random_state: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    interpolation: Optional[Interpolation] = None
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.interpolation is not None:
            X = self.interpolation(X)  # type: ignore

        optics = DBSCAN(metric=lambda x, y: self.metric(x, y, **self.algorithm_args), n_jobs=self.n_jobs)
        y: np.ndarray = optics.fit_predict(X)

        return y
