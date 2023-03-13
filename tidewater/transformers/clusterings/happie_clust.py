from typing import Any, Callable, Dict, Optional, List, Type, NamedTuple, Tuple
import warnings
from joblib import Parallel, delayed
import tqdm

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import estimate_bandwidth
from sklearn.utils import gen_batches
from sklearn.neighbors import KDTree
from tslearn.clustering import TimeSeriesKMeans
from pydantic.dataclasses import dataclass
from dataclasses import field
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from tslearn.metrics import dtw, soft_dtw
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import networkx as nx
import seaborn as sns
from numba import njit

from .base import DynSizeClustering
from ...datatypes import TimeSeries, Labels
from ..interface import InputInterface, InterfaceValue
from .distance_metrics import DistanceMetric, Interpolation
from .hierarchical import Hierarchical

import matplotlib.pyplot as plt


@njit
def _recalculate_distance(method: str, distance_a: float, distance_b: float) -> float:
    if method == "single":
        return min(distance_a, distance_b)
    elif method == "complete":
        return max(distance_a, distance_b)
    elif method == "average":
        return (distance_a + distance_b) / 2
    elif method == "weighted":
        return (distance_a + distance_b) / 2
    elif method == "centroid":
        return (distance_a + distance_b) / 2
    elif method == "median":
        return (distance_a + distance_b) / 2
    else:  # if method == "ward":
        return (distance_a + distance_b) / 2


def _approximate_hierarchical_clustering(X: List[np.ndarray], distances: nx.Graph, method: str) -> np.ndarray:
    n = len(X)
    # print(f"m = {len(distances.edges()) / ((n*(n-1))/2)}")
    z_matrix = []
    new_node = len(X)
    included_nodes = {i: 1 for i in range(len(X))}

    for _ in tqdm(range(n - 1)):
        (u, v, distance) = min(distances.edges(data="distance"), key=lambda x: x[2])
        # print(u, v, distance, included_nodes[u] + included_nodes[v])
        z_matrix.append([u, v, distance, included_nodes[u] + included_nodes[v]])
        # if u has more outgoing edges, swap u and v
        if len(distances[u]) > len(distances[v]):
            u, v = v, u

        new_edges = []
        stale_edges = []
        for (x, attr) in distances[u].items():
            if x == v:
                continue
            new_edge = (v, x)
            # check if edge already exists
            if x in distances[v]:
                distance = _recalculate_distance(method, attr["distance"], distances[v][x]["distance"])
                if distance is not None:
                    attr["distance"] = distance
                else:
                    raise ValueError(f"Unknown method: {method}")
                stale_edges.append((v, x))

            new_edges.append((*new_edge, attr))

        # update graph
        distances.remove_edges_from(stale_edges)
        distances.add_edges_from(new_edges)

        # rename node v to new_node to connect clusters
        distances = nx.relabel_nodes(distances, {v: new_node})
        included_nodes[new_node] = included_nodes[u] + included_nodes[v]
        distances.remove_node(u)
        new_node += 1
    return np.array(z_matrix)


@dataclass
class HappieClust(Hierarchical):
    n_pivots: int = 20
    s: float = 0.5
    m: float = 0.1

    def _choose_pivots(self, X: List[np.ndarray]) -> List[int]:
        pivots = np.random.choice(len(X), self.n_pivots, replace=False).tolist()
        return pivots

    def _estimate_epsilon(self, pivot_distances: np.ndarray) -> float:
        random_pairs = np.random.choice(pivot_distances.shape[0], (pivot_distances.shape[0], 2))
        random_pairs = self._remove_self_pairs(random_pairs)
        pseudo_distances = DistanceMetric.CHEBYSHEV.distance_pairs(
            pivot_distances,
            random_pairs,
            interpolation=self.interpolation,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.algorithm_args,
        )
        epsilon = np.quantile(pseudo_distances, self.s * self.m)

        return epsilon

    def _pairs_closer_than_epsilon(self, pivot_space: np.ndarray, epsilon: float) -> List[Tuple[int, int]]:
        kd_tree = KDTree(pivot_space, metric="chebyshev")
        neighbors = kd_tree.query_radius(pivot_space, epsilon)
        pairs = {(i, j) for i, n in enumerate(neighbors) if len(n) > 0 for j in n if i < j}
        return pairs

    def _generate_graph(self, X: List[np.ndarray], pivots: List[int]) -> nx.Graph:
        distance_graph = nx.Graph()
        distance_graph.add_nodes_from(range(len(X)))
        return distance_graph

    def _calculate_pivot_distances(self, X: List[np.ndarray], pivots: List[int]) -> np.ndarray:
        pivot_distances = self.metric.matrix_other(
            X,
            [X[p] for p in pivots],
            interpolation=self.interpolation,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.algorithm_args,
        )
        return pivot_distances

    def _calculate_distances_of_close_pairs(
        self, X: List[np.ndarray], pivot_distances: np.ndarray, distance_graph: nx.Graph
    ) -> nx.Graph:
        epsilon = self._estimate_epsilon(pivot_distances)
        close_pairs = self._pairs_closer_than_epsilon(pivot_distances, epsilon)
        close_distances = self.metric.distance_pairs(
            X,
            close_pairs,
            interpolation=self.interpolation,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.algorithm_args,
        )
        for (i, j), distance in zip(close_pairs, close_distances):
            distance_graph.add_edge(i, j, distance=distance)
        return distance_graph

    def _calculate_distances_of_random_pairs(self, X: List[np.ndarray], distance_graph: nx.Graph) -> nx.Graph:
        # calculate distances of additional random pairs
        n = len(X)
        m = int(self.m * (n * (n - 1)) / 2)
        random_pairs = np.random.choice(len(X), (int((1 - self.s) * m), 2))
        random_pairs = self._remove_self_pairs(random_pairs)
        random_distances = self.metric.distance_pairs(
            X,
            random_pairs,
            interpolation=self.interpolation,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.algorithm_args,
        )
        distance_graph.remove_edges_from(random_pairs)
        for (u, v), distance in zip(random_pairs, random_distances):
            distance_graph.add_edge(u, v, distance=distance)
        return distance_graph

    def _calculate_linkings(self, X: List[np.ndarray]) -> np.ndarray:
        pivots = self._choose_pivots(X)
        distance_graph = self._generate_graph(X, pivots)
        pivot_distances = self._calculate_pivot_distances(X, pivots)
        # print(f"edges: {distance_graph.number_of_edges()} after generation | {distance_graph.number_of_edges() / (len(X)*(len(X)-1)/2)}")
        distance_graph = self._calculate_distances_of_close_pairs(X, pivot_distances, distance_graph)
        # print(f"edges: {distance_graph.number_of_edges()} after close pairs | {distance_graph.number_of_edges() / (len(X)*(len(X)-1)/2)}")
        distance_graph = self._calculate_distances_of_random_pairs(X, distance_graph)
        # print(f"edges: {distance_graph.number_of_edges()} after random pairs | {distance_graph.number_of_edges() / (len(X)*(len(X)-1)/2)}")

        z_matrix = _approximate_hierarchical_clustering(X, distance_graph, self.method)
        return z_matrix

    def _remove_self_pairs(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(i, j) for i, j in pairs if i != j]


if __name__ == "__main__":
    clustering = HappieClust(metric=DistanceMetric.SHAPE_BASED_DISTANCE, n_clusters=2, method="average")
    z_matrix = clustering._calculate_linkings([np.random.rand(10) for _ in range(100)])
    print(z_matrix.shape)
    dendrogram(z_matrix)
    plt.show()
