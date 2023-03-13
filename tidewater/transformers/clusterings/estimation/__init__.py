from __future__ import annotations

from typing import Any, Optional, List, Dict, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm, trange

from tidewater.datatypes.timeseries import TimeSeries
from tidewater.transformers.interface import InputInterface

from ..distance_metrics import DistanceMetric
from ..birch import Birch
from ...encoders.birch_space_constructor import BirchSpaceConstructor
from ...encoders.base import Encoder
from ..base import DynSizeClustering, Clustering


def mean_sbd_minimize(a: List[np.ndarray]) -> np.ndarray:
    """Calculate the mean subsequence such that the shape based distance is minimized."""
    metric = DistanceMetric.SHAPE_BASED_DISTANCE
    history = []

    def f(x: np.ndarray) -> float:
        dist = sum([metric(a_, x) for a_ in a])
        history.append((x, dist))
        return dist

    min_len = min([a_.shape[0] for a_ in a])
    res = minimize(f, x0=(sum([a_[:min_len] for a_ in a]),), method="Nelder-Mead")
    print("Optimization finished with status", res.success, "and cost", res.fun)

    return res.x


@dataclass
class SubClusterLabels:
    ids: np.ndarray
    label: int
    hierarchy: int = 0
    sub_labels: List[int] = field(default_factory=list)


@dataclass
class PreClustering(DynSizeClustering):
    pre_clustering: Optional[Clustering] = None
    encoder: Optional[Encoder] = None
    metric: DistanceMetric = DistanceMetric.SHAPE_BASED_DISTANCE
    n_clusters: int = 2
    cluster_size_factor: int = 10
    n_jobs: int = 1
    verbose: bool = False
    plot: bool = False
    skip_dendrogram: bool = False
    algorithm_args: Dict[str, Any] = field(default_factory=lambda: {})
    mean_method: str = "sum_max_len"
    variance_estimation: bool = False
    birch_threshold: float = 0.1
    embedding_dimensions: int = 30
    embedding_choose_method: str = "median_of_median"
    sub_cluster_labels: List[SubClusterLabels] = field(default_factory=list)
    real_estimation: bool = False

    def __post_init__(self) -> None:
        super().__init__()

        if self.pre_clustering is None:
            self.pre_clustering = self._build_pre_clustering()
        if self.encoder is None:
            self.encoder = self._build_encoder()

    def _build_pre_clustering(self) -> Clustering:
        return Birch(n_clusters=self.n_clusters * self.cluster_size_factor, threshold=self.birch_threshold)

    def _build_encoder(self) -> Encoder:
        return BirchSpaceConstructor(
            n_clusters=self.n_clusters,
            cluster_size_factor=self.cluster_size_factor,
            threshold=self.birch_threshold,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            embedding_dimensions=self.embedding_dimensions,
            choose_method=self.embedding_choose_method,
            mean_method=self.mean_method,
        )

    def _pre_cluster(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        del kwargs["data"]
        encoded = self._get_encoded(X, **kwargs)
        pre_cluster_labels = self._pre_cluster_call(encoded, **kwargs)
        for i in np.unique(pre_cluster_labels):
            cluster_ids = np.where(pre_cluster_labels == i)[0]
            self.sub_cluster_labels.append(SubClusterLabels(cluster_ids, i))

        if not self.skip_dendrogram and False:
            counts_threshold = self.n_clusters * self.cluster_size_factor
            sub_clusters_to_check: List[SubClusterLabels] = [
                scl for scl in self.sub_cluster_labels if scl.ids.shape[0] > counts_threshold
            ]

            while len(sub_clusters_to_check) > 0:
                scl = sub_clusters_to_check.pop(0)
                sub_cluster_points = encoded[scl.ids]
                sub_labels = self._pre_cluster_call(sub_cluster_points, **kwargs)
                for i in np.unique(sub_labels):
                    cluster_ids = np.where(sub_labels == i)[0]
                    list_id = len(self.sub_cluster_labels)
                    self.sub_cluster_labels.append(SubClusterLabels(cluster_ids, i, scl.hierarchy + 1))
                    scl.sub_labels.append(list_id)

        return pre_cluster_labels

    def _get_encoded(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        if self.encoder is not None:
            encoded: np.ndarray = self.encoder._encode(X, n_jobs=self.n_jobs, verbose=self.verbose, **kwargs)
        else:
            encoded_timeseries = self.get_input_value("encoded")[0]
            assert (
                encoded_timeseries is not None
                and isinstance(encoded_timeseries, list)
                and isinstance(encoded_timeseries[0], TimeSeries)
            ), "Encoded timeseries not found"
            encoded = np.array([e.ndarray for e in encoded_timeseries])
        return encoded

    def _pre_cluster_call(self, encoded: np.ndarray, **kwargs) -> np.ndarray:
        if isinstance(self.pre_clustering, DynSizeClustering):
            pre_cluster_labels = self.pre_clustering._cluster_transform(encoded, **kwargs)  # type: ignore
        elif isinstance(self.pre_clustering, Clustering):
            pre_cluster_labels = self.pre_clustering._transform(encoded, **kwargs)
        else:
            raise ValueError(f"Unknown pre clustering type {type(self.pre_clustering)}")
        return pre_cluster_labels

    def _sub_cluster(self, X: List[np.ndarray], **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) > 1:
            distances = self.metric.z_matrix(X, n_jobs=self.n_jobs, **kwargs)
            linkage_matrix = linkage(distances, method="ward")
        else:
            linkage_matrix = np.array([[0, 0, 0, 1]])
            distances = np.array([[0]])
        return linkage_matrix, distances

    def _combine_linkages(
        self, X: List[np.ndarray], pre_clustering_results: List[PreClusteringResult], **kwargs
    ) -> np.ndarray:
        if not self.variance_estimation:
            # 1. Aggregate each sub cluster into a single data point
            aggregates = [
                self.metric.aggregate(c.points, method=self.mean_method, distance_matrix=c.distances)
                for c in pre_clustering_results
            ]

            # 2. Compute the distance between each sub cluster
            distances = self.metric.z_matrix(
                aggregates, n_jobs=self.n_jobs, verbose=self.verbose, **self.algorithm_args, **kwargs
            )
            # 3. Compute the linkage between each sub cluster
            linkage_matrix = linkage(distances, method="ward")
        else:
            # 1. Estimate the variance of each sub cluster
            linkage_matrix = self._linkage_estimation(X, pre_clustering_results, **kwargs)
            # print(linkage_matrix)

        # 4. Combine the linkage matrices
        if len(pre_clustering_results) > 0:
            warnings.warn("The linking of dendrograms is not yet implemented for pre-clustering")

        # 5. Return the linkage matrix
        return linkage_matrix

    def _linkage_estimation(self, X: List[np.ndarray], pcr: List[PreClusteringResult], **kwargs: Any) -> np.ndarray:
        linkage_matrix: List[Tuple[int, int, float, int]] = []
        if self.real_estimation:
            distance_matrix = self.metric.matrix(
                X, verbose=self.verbose, n_jobs=self.n_jobs, **self.algorithm_args, **kwargs
            )
        else:
            distance_matrix = None
        variances = VarianceMemory(pcr, real=self.real_estimation, _distance_matrix=distance_matrix)

        last_min_value = 0.0
        for i in trange(len(pcr) - 1, desc="Linkage estimation", disable=not self.verbose):
            min_pair = variances.argmin()
            min_value: float = variances[min_pair]
            variances.add_cluster(min_pair)
            last_min_value += min_value
            linkage_matrix.append((*min_pair, last_min_value, variances.n_obserivations(len(pcr) + i)))

        return np.array(linkage_matrix).astype(float)

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        pre_clusters = self._pre_cluster(X, **kwargs)
        pre_clustering_results = []
        for c, scl in tqdm(
            enumerate(self.sub_cluster_labels), desc="Sub-clustering", total=len(self.sub_cluster_labels)
        ):
            if len(scl.sub_labels) == 0:
                points = [X[i] for i in scl.ids]
                if not self.skip_dendrogram:
                    linkage_matrix, distances = self._sub_cluster(points, **kwargs)
                else:
                    linkage_matrix, distances = None, None
                pre_clustering_results.append(
                    PreClusteringResult(
                        c,
                        points=points,
                        _point_ids=scl.ids,
                        linkage_matrix=linkage_matrix,
                        distances=distances,
                        metric=self.metric,
                    )
                )
        linkage_matrix = self._combine_linkages(X, pre_clustering_results)
        if self.plot:
            self._plot(linkage_matrix)
        labels = cut_tree(linkage_matrix, n_clusters=self.n_clusters).reshape(-1)
        return np.array([labels[c] for c in pre_clusters])

    def _plot(self, X: np.ndarray, **kwargs: Any) -> None:
        dendrogram(X, **kwargs)
        plt.show()


class PreClusteringNoEncoding(PreClustering):
    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], encoded=Optional[List[TimeSeries]])  # type: ignore


@dataclass
class PreClusteringResult:
    scl_id: int
    points: List[np.ndarray]
    linkage_matrix: Optional[np.ndarray] = None
    distances: Optional[np.ndarray] = None
    metric: Optional[DistanceMetric] = None
    _point_ids: Optional[np.ndarray] = None

    @property
    def var(self) -> float:
        return np.mean(self.distances)

    @property
    def medoid(self) -> np.ndarray:
        return self.points[np.argmin(self.distances.sum(axis=0))]

    def __len__(self) -> int:
        return len(self.points)

    def __sub__(self, other: PreClusteringResult) -> float:
        medoid_distance = self.metric(self.medoid, other.medoid)
        estimated_variance = (
            self.var * (len(self) ** 2) + other.var * (len(other) ** 2) + (medoid_distance) * (len(self) * len(other))
        ) / (len(self) ** 2 + len(other) ** 2 + len(self) * len(other))
        return estimated_variance

    @staticmethod
    def estimate_new_variance(*pcrs: PreClusteringResult, medoid_distances) -> float:
        old_variances = sum(pcr.var * (len(pcr) ** 2) for pcr in pcrs)
        distances = sum(
            (medoid_distances[i, j]) * (len(pcrs[i]) * len(pcrs[j]))
            for i in range(len(pcrs))
            for j in range(i + 1, len(pcrs))
        )
        normalizer = sum(len(pcr) ** 2 for pcr in pcrs) + sum(
            len(pcrs[i]) * len(pcrs[j]) for i in range(len(pcrs)) for j in range(i + 1, len(pcrs))
        )

        return (old_variances + distances) / normalizer


@dataclass
class VarianceMemory:
    pre_clustering_results: List[PreClusteringResult]
    real: bool = False
    _medoid_distances: Optional[np.ndarray] = None
    _variance_changes: List[Dict[int, float]] = field(default_factory=list)
    _clusters: List[List[int]] = field(default_factory=list)
    _remaining: Set[int] = field(default_factory=set)
    _cluster_variances: List[float] = field(default_factory=list)
    _distance_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        l = len(self.pre_clustering_results)
        p_bar = tqdm(total=int((l**2 - l) / 2), desc="Calculating variance changes")
        for i in range(l):
            c1 = self.pre_clustering_results[i]
            c1_var = c1.var
            self._cluster_variances.append(c1_var)
            for j in range(i, l):
                if i == j:
                    distance = np.inf
                else:
                    c2 = self.pre_clustering_results[j]
                    if self.real:
                        distance = self._distance_matrix[np.concatenate((c1._point_ids, c2._point_ids))][
                            :, np.concatenate((c1._point_ids, c2._point_ids))
                        ].mean()
                    else:
                        estimated_variance = c1 - c2
                        distance = abs(c1_var - estimated_variance) + abs(c2.var - estimated_variance)
                self[i, j] = distance
                p_bar.update(1)
        p_bar.close()

        # Calculate medoid distances
        metric = self.pre_clustering_results[0].metric
        assert metric is not None, "Metric must be set"
        medoids = [pcr.medoid for pcr in self.pre_clustering_results]
        self._medoid_distances = metric.matrix(medoids)

    def _order_keys(self, key: Tuple[int, int]) -> Tuple[int, int]:
        i, j = key
        if i > j:
            return j, i
        return i, j

    def _next_id(self) -> int:
        return len(self._cluster_variances)

    def _get_cluster_elements(self, cluster_id: int) -> List[int]:
        if cluster_id < len(self.pre_clustering_results):
            return [cluster_id]
        else:
            return self._clusters[cluster_id - len(self.pre_clustering_results)]

    def _get_cluster_variance(self, cluster_id: int) -> float:
        return self._cluster_variances[cluster_id]

    def _recalculate_variance_changes(self, cluster_id: int) -> None:
        cluster_elements = self._get_cluster_elements(cluster_id)
        variance = self._get_cluster_variance(cluster_id)
        for i in self._remaining:
            if i == cluster_id:
                continue
            cluster_elements_i = self._get_cluster_elements(i)
            variance_i = self._get_cluster_variance(i)
            if self.real:
                point_ids = np.concatenate(
                    [self.pre_clustering_results[j]._point_ids for j in cluster_elements + cluster_elements_i]
                )
                new_variance = self._distance_matrix[point_ids][:, point_ids].mean()
            else:
                new_variance = PreClusteringResult.estimate_new_variance(
                    *[self.pre_clustering_results[j] for j in cluster_elements + cluster_elements_i],
                    medoid_distances=self._medoid_distances,
                )

            variance_distance = abs(variance - new_variance) + abs(variance_i - new_variance)

            self[cluster_id, i] = variance_distance
            if len(self._variance_changes) <= cluster_id:
                assert len(self._variance_changes) == cluster_id, "Variance changes must be in order"
                self._variance_changes.append(defaultdict(lambda: np.nan))

    def _remove_old_cluster(self, cluster_id: int) -> None:
        self._variance_changes[cluster_id] = {k: np.inf for k in self._variance_changes[cluster_id]}
        for i in self._variance_changes:
            if cluster_id in i:
                i[cluster_id] = np.inf
        self._remaining.remove(cluster_id)

    def __getitem__(self, key: Any) -> Union[float, np.ndarray]:
        if type(key) == tuple:
            i, j = self._order_keys(key)
            return self._variance_changes[i][j]
        elif type(key) == int:
            variances_larger = self._variance_changes[key]
            variances_smaller = [self._variance_changes[j][key] for j in range(key)]
            return np.array(variances_smaller + [0.0] + list(variances_larger.values()))
        else:
            raise TypeError("Key must be of type tuple or int")

    def __setitem__(self, key: Tuple[int, int], value: float) -> None:
        i, j = self._order_keys(key)
        if i >= len(self._variance_changes):
            for _ in range(i - len(self._variance_changes) + 1):
                self._variance_changes.append(defaultdict(lambda: np.nan))
        self._variance_changes[i][j] = value
        self._remaining.add(i)
        self._remaining.add(j)

    def argmin(self) -> Tuple[int, int]:
        argmin: Optional[Tuple[int, int]] = None
        min_value = np.inf
        for i, d in enumerate(self._variance_changes):
            for j in d:
                if i == j:
                    continue
                if d[j] < min_value:
                    min_value = d[j]
                    argmin = (i, j)
        assert argmin is not None, "No minimum found"
        return argmin

    def add_cluster(self, key: Tuple[int, int]) -> None:
        i, j = self._order_keys(key)
        new_id = self._next_id()

        # Remove clustered elements from remaining and add new cluster
        self._remaining.add(new_id)
        self._remove_old_cluster(i)
        self._remove_old_cluster(j)

        cluster_elements = self._get_cluster_elements(i) + self._get_cluster_elements(j)
        self._clusters.append(cluster_elements)
        if self.real:
            point_ids = np.concatenate([self.pre_clustering_results[j]._point_ids for j in cluster_elements])
            self._cluster_variances.append(self._distance_matrix[point_ids][:, point_ids].mean())
        else:
            self._cluster_variances.append(
                PreClusteringResult.estimate_new_variance(
                    *[self.pre_clustering_results[j] for j in cluster_elements], medoid_distances=self._medoid_distances
                )
            )
        self._recalculate_variance_changes(new_id)

    def n_obserivations(self, cluster_id: int) -> int:
        return len(self._get_cluster_elements(cluster_id))
