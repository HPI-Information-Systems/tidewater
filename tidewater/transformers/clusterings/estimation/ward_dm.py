from ast import Tuple
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from tqdm import trange
from scipy.cluster.hierarchy import cut_tree, ward
import code

from ....datatypes import TimeSeries, Labels
from ...interface import InputInterface
from ..distance_metrics import DistanceMetric
from ..base import DynSizeClustering


@dataclass
class WardDM(DynSizeClustering):
    n_clusters: int = 3
    metric: DistanceMetric = DistanceMetric.SHAPE_BASED_DISTANCE
    estimation: bool = False
    verbose: bool = False
    n_jobs: int = 1
    _distance_matrix: Optional[np.ndarray] = None
    _distance_matrices: List[np.ndarray] = field(default_factory=list)
    _cluster_indices: List[np.ndarray] = field(default_factory=list)
    _medoid_distances: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        super().__init__()

    def _get_labels(self) -> np.ndarray:
        labels = self.get_input_value("labels")[0]
        assert labels is not None and isinstance(labels, Labels), "Labels must be provided"
        return labels.ndarray

    def _build_distance_matrix(self, X: List[np.ndarray], labels: np.ndarray):
        self._distance_matrix = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            i_label = labels[i]
            for j in range(i + 1, len(X)):
                j_label = labels[j]
                if i_label == j_label:
                    pseudo_i, pseudo_j = (
                        np.where(self._cluster_indices[i_label] == i)[0][0],
                        np.where(self._cluster_indices[i_label] == j)[0][0],
                    )
                    distance = self._distance_matrices[i_label][pseudo_i, pseudo_j]
                else:
                    distance = self._medoid_distances[i_label, j_label]
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance

    def _random_validity_check(self, n: int, labels: np.ndarray):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        while labels[i] == labels[j]:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
        d = self._distance_matrix[i, j]
        medoid_d = self._medoid_distances[labels[i], labels[j]]

        assert d == medoid_d, "Distance matrix is not correct"

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        labels = self._get_labels()
        medoids = []
        for c in np.unique(labels):
            ids = np.where(labels == c)[0]
            dm = self.metric.matrix([X[i] for i in ids])
            self._distance_matrices.append(dm)
            self._cluster_indices.append(ids)
            medoid = np.argmin(dm.sum(axis=0))
            medoids.append(ids[medoid])
        self._medoid_distances = self.metric.matrix([X[i] for i in medoids], n_jobs=self.n_jobs)
        self._build_distance_matrix(X, labels)
        self._random_validity_check(len(X), labels)

        condensed_distance_matrix = self._distance_matrix[np.triu_indices(len(X), k=1)]
        Z = ward(condensed_distance_matrix)

        cluster_labes = cut_tree(Z, n_clusters=self.n_clusters).reshape(-1)
        return cluster_labes

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels)
