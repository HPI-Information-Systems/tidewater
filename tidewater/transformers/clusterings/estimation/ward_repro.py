from ast import Tuple
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from tqdm import trange
from scipy.cluster.hierarchy import cut_tree, ward
import math

from ....datatypes import TimeSeries, Labels
from ...interface import InputInterface
from ..distance_metrics import DistanceMetric
from ..base import DynSizeClustering
import code


def ward_fn(d_xi: float, d_yi: float, d_xy: float, size_x: int, size_y: int, size_i: int):
    t = 1.0 / (size_x + size_y + size_i)
    return math.sqrt(
        (size_i + size_x) * t * d_xi * d_xi + (size_i + size_y) * t * d_yi * d_yi - size_i * t * d_xy * d_xy
    )


def condensed_index(n: int, i: int, j: int) -> int:
    if i > j:
        i, j = j, i
    return int(n * i - i * (i + 1) / 2 + j - i - 1)


def nn_chain(dists: np.ndarray, n: int) -> np.ndarray:
    """Perform hierarchy clustering using nearest-neighbor chain algorithm.
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    Z_arr = np.empty((n - 1, 4))
    Z = Z_arr

    D = dists.copy()  # Distances between clusters.
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    for k in range(n - 1):
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]

            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = np.inf

            for i in range(n):
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            D[condensed_index(n, i, y)] = ward_fn(
                D[condensed_index(n, i, x)], D[condensed_index(n, i, y)], current_min, nx, ny, ni
            )

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind="mergesort")
    Z_arr = Z_arr[order]

    parents = {}
    new_id = len(Z_arr) + 1
    for z in Z_arr:
        x, y = int(z[0]), int(z[1])

        if x in parents:
            z[0] = parents[x]

        if y in parents:
            z[1] = parents[y]

        parents[x] = new_id
        parents[y] = new_id
        new_id += 1

    return Z_arr


@dataclass
class Ward(DynSizeClustering):
    n_clusters: int = 3
    metric: DistanceMetric = DistanceMetric.SHAPE_BASED_DISTANCE
    estimation: bool = False
    verbose: bool = False
    n_jobs: int = 1
    _initial_cluster_number: int = 0
    _distance_matrix: Optional[np.ndarray] = None
    _clusters: List[np.ndarray] = field(default_factory=list)  # List of subsequence ids in each cluster
    _pseudo_cluster_sizes: List[int] = field(default_factory=list)  # List of element number in each pseudo-cluster
    _variances: List[float] = field(default_factory=list)
    _ignore_ids: List[int] = field(default_factory=list)
    _z_matrix: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    def _get_labels(self) -> np.ndarray:
        labels = self.get_input_value("labels")[0]
        assert labels is not None and isinstance(labels, Labels), "Labels must be provided"
        return labels.ndarray

    def _cluster_variance(self, ids: np.ndarray) -> float:
        return np.mean(self._distance_matrix[ids][:, ids])

    def _calculate_variance_changes(self) -> np.ndarray:
        variance_changes = np.zeros((len(self._clusters), len(self._clusters))) + np.inf
        for i in range(len(self._clusters)):
            if i in self._ignore_ids:
                continue
            for j in range(i + 1, len(self._clusters)):
                if j in self._ignore_ids:
                    continue
                ids = np.concatenate((self._clusters[i], self._clusters[j]))
                var_i = self._variances[i]
                var_j = self._variances[j]
                new_var = self._cluster_variance(ids)
                # if new_var < var_i or new_var < var_j:
                #     code.interact(local=locals())
                variance_change = (
                    (new_var - var_i) * self._get_cluster_size(i) + (new_var - var_j) * self._get_cluster_size(j)
                ) / (self._get_cluster_size(i) + self._get_cluster_size(j))
                variance_changes[i, j] = variance_change
                variance_changes[j, i] = variance_change
        return variance_changes

    def _add_cluster(self, ids: np.ndarray) -> None:
        self._clusters.append(ids)
        self._variances.append(self._cluster_variance(ids))

    def _get_cluster_size(self, cluster_id: int) -> int:
        return len(self._clusters[cluster_id])

    def _get_pseudo_cluster_size(self, cluster_id: int) -> int:
        return self._pseudo_cluster_sizes[cluster_id]

    def _perform_clustering(self, X: List[np.ndarray]) -> None:
        for _ in trange(len(self._clusters) - 1, desc="Clustering", disable=not self.verbose):
            variance_changes = self._calculate_variance_changes()
            (i, j) = np.unravel_index(np.argmin(variance_changes), variance_changes.shape)
            self._ignore_ids.append(i)
            self._ignore_ids.append(j)

            ids = np.concatenate((self._clusters[i], self._clusters[j]))
            self._add_cluster(ids)
            self._pseudo_cluster_sizes.append(self._get_pseudo_cluster_size(i) + self._get_pseudo_cluster_size(j))
            self._z_matrix.append([i, j, variance_changes[i, j], self._pseudo_cluster_sizes[-1]])

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        labels = self._get_labels()
        try:
            self._distance_matrix = np.load("distance_matrix.npy")
        except:
            self._distance_matrix = self.metric.matrix(X, verbose=self.verbose, n_jobs=self.n_jobs)
            np.save("distance_matrix.npy", self._distance_matrix)

        for c in np.unique(labels):
            ids = np.where(labels == c)[0]
            self._add_cluster(ids)
            self._pseudo_cluster_sizes.append(1)
        self._initial_cluster_number = len(self._clusters)

        # self._perform_clustering(X)
        condensed_distance_matrix = self._distance_matrix[np.triu_indices(len(self._distance_matrix), k=1)]
        self._z_matrix = nn_chain(condensed_distance_matrix, len(self._distance_matrix))

        return cut_tree(self._z_matrix, n_clusters=self.n_clusters).reshape(-1)

        print(self._z_matrix)
        Z = np.array(self._z_matrix)
        # sort_indices = np.argsort(Z[:, 2])
        # self._z_matrix = Z[sort_indices]

        cluster_labes = cut_tree(Z, n_clusters=self.n_clusters).reshape(-1)
        print(cluster_labes, len(labels), len(cluster_labes))
        return np.array([cluster_labes[c] for c in labels])

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels)


if __name__ == "__main__":
    distance_matrix = np.random.rand(10, 10)
    condensed_distance_matrix = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    print(distance_matrix[5, 6])
    print(condensed_index(10, 5, 6), condensed_distance_matrix[condensed_index(10, 5, 6)])
    assert condensed_distance_matrix[condensed_index(10, 5, 6)] == distance_matrix[5, 6]
