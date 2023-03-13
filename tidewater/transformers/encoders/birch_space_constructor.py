from typing import Any, Optional, List

import warnings
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import Birch as SLBirch
from sklearn.exceptions import ConvergenceWarning
from dataclasses import dataclass, field
from tslearn.utils import to_time_series_dataset
import tqdm
from enum import Enum
import matplotlib.pyplot as plt

from .base import Encoder
from ..clusterings.distance_metrics import DistanceMetric, Interpolation


class ChooseMethod(Enum):
    RANDOM = "random"
    MEDIAN_OF_MEDIAN = "median_of_median"
    MEDIAN_OF_RANDOM = "median_of_random"
    RANDOM_CLOSEST_FARTHEST = "random_closest_farthest"
    RANDOM_LARGEST = "random_largest"
    RANDOM_WEIGHTED = "random_weighted"


@dataclass
class BirchSpaceConstructor(Encoder):
    metric: DistanceMetric = DistanceMetric.SHAPE_BASED_DISTANCE
    n_clusters: int = 3
    threshold: float = 0.1
    branching_factor: int = 50
    random_state: Optional[int] = None
    n_jobs: int = 1
    embedding_dimensions: int = 30
    initial_embedding_indices: Optional[List[int]] = None
    choose_method: ChooseMethod = ChooseMethod.RANDOM_LARGEST
    mean_method: str = "sum_max_len"
    verbose: bool = False
    random_set_size: int = 5
    _embedding_space: Optional[np.ndarray] = None
    _embedding_indices: List[int] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        super().__init__()
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def _birching(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        model = SLBirch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor,
        )
        labels: np.ndarray = model.fit_predict(X)
        return labels

    def _add_embedding_dimension(self, indices: List[int], data: List[np.ndarray]) -> None:
        self._embedding_space = np.c_[
            self._embedding_space, self.metric.matrix_other(data, [data[i] for i in indices], n_jobs=self.n_jobs)
        ]
        self._embedding_indices.extend(indices)

    def _calculate_centroid(self, data: List[np.ndarray]) -> np.ndarray:
        min_len = min([len(d) for d in data])
        return np.sum([d[:min_len] for d in data], axis=0)

    def _choose_random(self, closest_centroid_id: int, labels: np.ndarray) -> List[int]:
        return [np.random.choice(np.where(labels == closest_centroid_id)[0])]

    def _choose_median_of_median(
        self,
        closest_centroid_id: int,
        farthest_centroid_id: int,
        labels: np.ndarray,
        data: List[np.ndarray],
        centroids: List[np.ndarray],
    ) -> List[int]:
        # take points from the closest centroid and calculate the distances to the farthest centroid
        closest_centroid_points = [data[i] for i in np.where(labels == closest_centroid_id)[0]]
        closest_centroid_distances = self.metric.matrix_other(
            closest_centroid_points, [centroids[farthest_centroid_id]], n_jobs=self.n_jobs
        )
        # print(f"closest centroid points: {np.where(labels == closest_centroid_id)[0]}")

        # take points from the closest centroid that are the farthest from the farthest centroid
        return [np.where(labels == closest_centroid_id)[0][closest_centroid_distances.argmin()]]

    def _choose_median_of_random(
        self,
        closest_centroid_id: int,
        farthest_centroid_id: int,
        labels: np.ndarray,
        data: List[np.ndarray],
        centroids: List[np.ndarray],
    ) -> List[int]:
        # take some random points from the closest centroid and calculate the distances to the farthest centroid
        centroid_elements = np.where(labels == closest_centroid_id)[0]
        sample_size = min(self.random_set_size, len(centroid_elements))
        random_points = np.random.choice(centroid_elements, size=sample_size, replace=False)
        random_points_distances = self.metric.matrix_other(
            [data[i] for i in random_points], [centroids[farthest_centroid_id]], n_jobs=self.n_jobs
        )

        # take the point from the closest centroid that is the farthest from the farthest centroid
        return [random_points[random_points_distances.argmin()]]

    def _choose_random_closest_farthest(
        self, closest_centroid_id: int, farthest_centroid_id: int, labels: np.ndarray
    ) -> List[int]:
        # take a random point from the closest centroid
        closest_centroid_points = np.where(labels == closest_centroid_id)[0]
        random_point = np.random.choice(closest_centroid_points, size=1, replace=False)[0]

        # take a random point from the farthest centroid
        farthest_centroid_points = np.where(labels == farthest_centroid_id)[0]
        random_point2 = np.random.choice(farthest_centroid_points, size=1, replace=False)[0]

        return [random_point, random_point2]

    def _choose_random_largest(self, labels: np.ndarray) -> List[int]:
        # take a random point from the largest cluster
        largest_cluster = np.argmax(np.bincount(labels))
        largest_cluster_points = np.where(labels == largest_cluster)[0]
        largest_cluster_points_not_in_embedding = [
            p for p in largest_cluster_points if p not in self._embedding_indices
        ]

        if len(largest_cluster_points_not_in_embedding) == 0:
            return [self._embedding_indices]

        random_point = None
        while random_point is None or random_point in self._embedding_indices:
            random_point = np.random.choice(largest_cluster_points, size=1, replace=False)[0]

        # plot the distribution of clusters
        # plt.hist(labels, bins=np.unique(labels).shape[0])
        # plt.show()

        return [random_point]

    def _choose_random_weighted(self, labels: np.ndarray) -> List[int]:
        # sample from all points that are not in the embedding indices, weighted by the reciprocal of the cluster size
        cluster_sizes = np.bincount(labels)
        cluster_sizes = cluster_sizes.sum() / cluster_sizes
        points_not_in_embedding, weights = zip(
            *[(i, cluster_sizes[p]) for i, p in enumerate(labels) if i not in self._embedding_indices]
        )
        weights = weights / np.sum(weights)
        random_point = np.random.choice(points_not_in_embedding, size=1, replace=False, p=weights)[0]

        return [random_point]

    def _choose_next_dimension(
        self,
        data: List[np.ndarray],
        labels: np.ndarray,
        centroids: List[np.ndarray],
        closest_centroid_id: int,
        farthest_centroid_id: int,
    ) -> List[int]:
        if self.choose_method == ChooseMethod.RANDOM:
            new_dimension_index = self._choose_random(closest_centroid_id, labels)
        elif self.choose_method == ChooseMethod.MEDIAN_OF_MEDIAN:
            new_dimension_index = self._choose_median_of_median(
                closest_centroid_id, farthest_centroid_id, labels, data, centroids
            )
        elif self.choose_method == ChooseMethod.MEDIAN_OF_RANDOM:
            new_dimension_index = self._choose_median_of_random(
                closest_centroid_id, farthest_centroid_id, labels, data, centroids
            )
        elif self.choose_method == ChooseMethod.RANDOM_CLOSEST_FARTHEST:
            new_dimension_index = self._choose_random_closest_farthest(
                closest_centroid_id, farthest_centroid_id, labels
            )
        elif self.choose_method == ChooseMethod.RANDOM_LARGEST:
            new_dimension_index = self._choose_random_largest(labels)
        elif self.choose_method == ChooseMethod.RANDOM_WEIGHTED:
            new_dimension_index = self._choose_random_weighted(labels)
        else:
            raise ValueError(f"choose_method {self.choose_method} not supported")
        return new_dimension_index

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        """Call Birch in a small embedding space and iteratively add embedding dimensions until
        the embedding space has `embedding_dimensions` dimensions.
        A new embedding dimension is added if a randomly chosen pair of points from two different
        clusters are closer than their respective cluster centroids.
        """

        if self.initial_embedding_indices is not None:
            self._embedding_indices = self.initial_embedding_indices
            if self.verbose:
                print(f"using initial embedding indices: {len(self._embedding_indices)}")
        else:
            self._embedding_indices = np.random.choice(np.arange(len(data)), size=2, replace=False).tolist()

        self._embedding_space = self.metric.matrix_other(
            data, [data[i] for i in self._embedding_indices], n_jobs=self.n_jobs
        )

        p_bar = tqdm.tqdm(total=self.embedding_dimensions - 2, disable=not self.verbose, desc="embedding steps")
        while len(self._embedding_indices) < self.embedding_dimensions:
            labels = self._birching(self._embedding_space)
            # plt.hist(labels, bins=np.unique(labels).shape[0])
            # plt.show()

            centroids = [
                self.metric.aggregate([data[i] for i in np.where(labels == c)[0]], method=self.mean_method)
                for c in np.unique(labels)
            ]
            centroid_distances = self.metric.matrix(centroids, n_jobs=self.n_jobs)

            farthest_centroid_id = np.sum(centroid_distances, axis=1).argmax()
            closest_centroid_id = np.sum(centroid_distances, axis=1).argmin()
            new_dimension_index = self._choose_next_dimension(
                data, labels, centroids, closest_centroid_id, farthest_centroid_id
            )

            if any([i in self._embedding_indices for i in new_dimension_index]):
                break

            self._add_embedding_dimension(new_dimension_index, data)
            p_bar.update(1)
        p_bar.close()
        if self.verbose:
            print(f"embedding space has {len(self._embedding_indices)} dimensions")

        assert self._embedding_space is not None, "embedding space should not be None"
        return self._embedding_space
