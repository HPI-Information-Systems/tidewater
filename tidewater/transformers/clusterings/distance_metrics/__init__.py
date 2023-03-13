from __future__ import annotations
from enum import Enum
from joblib import Parallel, delayed
import numpy as np
from numpy import typing as npt
from typing import Any, Callable, Optional, Tuple, Union, List
from scipy.spatial.distance import pdist
from scipy.signal import correlate
from scipy.stats import pearsonr
from tslearn.metrics import lcss, dtw
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from .msm import move_split_merge
import matplotlib.pyplot as plt


class Interpolation(Enum):
    UP = 0
    DOWN = 1

    def _interp(self, series: Union[np.ndarray, List[np.ndarray]], target_len: int) -> List[np.ndarray]:
        return [np.interp(np.linspace(0, len(s) - 1, target_len), np.arange(len(s)), s) for s in series]

    def __call__(
        self, series: Union[np.ndarray, List[np.ndarray]], other: Union[np.ndarray, List[np.ndarray], None] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self == Interpolation.UP:
            target_len = max(len(x) for x in series)
            if other is not None:
                target_len = max(max(len(x) for x in other), target_len)
        else:  # if self == Interpolation.DOWN:
            target_len = min(len(x) for x in series)
            if other is not None:
                target_len = min(min(len(x) for x in other), target_len)

        interpolated = self._interp(series, target_len)

        if other is not None:
            interpolated_other = self._interp(other, target_len)
            return np.array(interpolated), np.array(interpolated_other)
        return np.array(interpolated)


class DistanceMetric(Enum):
    EUCLIDEAN = 0
    # AVERAGE = 1
    # WEIGHTED_EUCLIDEAN = 2
    MANHATTAN = 3
    LONGEST_COMMON_SUBSEQUENCE = 4
    DTW = 5
    # SPADE = 6
    CROSS_CORRELATION = 8
    # SAX = 9
    # CHORD = 10
    # MAHALANOBIS = 11
    COSINE = 12
    # MEAN_CHARACTER_DIFFERENCE = 13
    # INDEX_OF_ASSOCIATION = 14
    CANBERRA = 15
    # CZEKANOWSKI_COEFFICIENT = 16
    # COEFFICIENT_OF_DIVERGENCE = 17
    PEARSON_COEFFICIENT = 18
    SHAPE_BASED_DISTANCE = 19
    # STANDARDIZED_EUCLIDEAN = 20
    CORRELATION = 21
    HAMMING = 22
    JACCARD = 23
    CHEBYSHEV = 24
    BRAYCURTIS = 25
    MSM = 26

    def _get_dist_fn(self, **kwargs: Any) -> Callable[[Any, Any], float]:
        if self == self.EUCLIDEAN:
            return lambda x, y: float(pdist([x, y], "euclidean").item(0))

        # if self == self.AVERAGE:
        #     raise NotImplementedError()

        # if self == self.WEIGHTED_EUCLIDEAN:
        #     raise NotImplementedError()

        if self == self.MANHATTAN:
            return lambda x, y: float(pdist([x, y], "cityblock").item(0))

        if self == self.LONGEST_COMMON_SUBSEQUENCE:
            return lambda x, y: float(1 - lcss(x, y))

        if self == self.DTW:
            return lambda x, y: float(dtw(x, y))

        # if self == self.SPADE:
        #     raise NotImplementedError()

        if self == self.CROSS_CORRELATION:
            return lambda x, y: float(1 - (np.argmax(correlate(x, y, method="fft") / len(x)) + 1 - len(x)))

        # if self == self.SAX:
        #     raise NotImplementedError()

        # if self == self.CHORD:
        #     raise NotImplementedError()

        # if self == self.MAHALANOBIS:
        #     return lambda x, y: float(pdist([x, y], "mahalanobis").item(0))

        if self == self.COSINE:
            return lambda x, y: float(pdist([x, y], "cosine").item(0))

        # if self == self.MEAN_CHARACTER_DIFFERENCE:
        #     raise NotImplementedError()

        if self == self.CANBERRA:
            return lambda x, y: float(pdist([x, y], "canberra").item(0))

        if self == self.PEARSON_COEFFICIENT:
            return lambda x, y: float(pdist([x, y], metric=lambda x, y: 1 - abs(pearsonr(x, y)[0])).item(0))

        if self == self.SHAPE_BASED_DISTANCE:
            return lambda x, y: abs(
                float(1 - np.max(correlate(x, y, method="fft") / np.sqrt(np.dot(x, x) * np.dot(y, y))))
            )

        # if self == self.STANDARDIZED_EUCLIDEAN:
        #     return lambda x, y: float(pdist([x, y], "seuclidean").item(0))

        if self == self.CORRELATION:
            return lambda x, y: float(pdist([x, y], "correlation").item(0))

        if self == self.HAMMING:
            return lambda x, y: float(pdist([x, y], "hamming").item(0))

        if self == self.JACCARD:
            return lambda x, y: float(pdist([x, y], "jaccard").item(0))

        if self == self.CHEBYSHEV:
            return lambda x, y: float(pdist([x, y], "chebyshev").item(0))

        if self == self.BRAYCURTIS:
            return lambda x, y: float(pdist([x, y], "braycurtis").item(0))

        if self == self.MSM:
            c = float(kwargs.get("c", 0.0))
            try:
                from move_split_merge import MoveSplitMerge

                return lambda x, y: float(MoveSplitMerge.PAPER(x, y, constant=c))
            except ImportError:
                return lambda x, y: float(move_split_merge(x, y, constant=c))

        raise ValueError("This distance measure is not known")

    def distance_pairs(
        self,
        series: Union[np.ndarray, List[np.ndarray]],
        pairs: Union[List[Tuple[int, int]], np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        n_jobs = kwargs.get("n_jobs", 1)
        with tqdm_joblib(tqdm(desc=self.name, total=len(pairs), disable=not kwargs.get("verbose", False))):
            distances = Parallel(n_jobs=n_jobs)(
                delayed(self._get_dist_fn(**kwargs))(series[i], series[j]) for i, j in pairs  # only 1d metrics so far
            )
        return np.array(distances)

    def z_matrix(
        self, series: Union[np.ndarray, List[np.ndarray]], interpolation: Optional[Interpolation] = None, **kwargs: Any
    ) -> np.ndarray:
        if interpolation is not None:
            series = interpolation(series)  # type: ignore
        self._validate_series_lengths(series)
        n_jobs = kwargs.get("n_jobs", 1)
        with tqdm_joblib(
            tqdm(desc=self.name, total=((len(series) ** 2) - len(series)) / 2, disable=not kwargs.get("verbose", False))
        ):
            z = Parallel(n_jobs=n_jobs)(
                delayed(self._get_dist_fn(**kwargs))(series[i], series[j])  # only 1d metrics so far
                for i in range(len(series) - 1)
                for j in range(i + 1, len(series))
            )
        return np.array(z).flatten()

    def matrix(
        self, series: Union[np.ndarray, List[np.ndarray]], interpolation: Optional[Interpolation] = None, **kwargs: Any
    ) -> np.ndarray:
        len_s = len(series)
        z = self.z_matrix(series, interpolation=interpolation, **kwargs)
        distance_matrix = np.zeros((len_s, len_s))
        last_end = 0
        for i in range(len_s):
            values = z[last_end : last_end + len_s - (i + 1)]
            distance_matrix[i, i + 1 :] = values
            distance_matrix[i + 1 :, i] = values
            last_end += len_s - (i + 1)
        return distance_matrix

    def matrix_other(
        self,
        series: Union[np.ndarray, List[np.ndarray]],
        other: Union[np.ndarray, List[np.ndarray]],
        interpolation: Optional[Interpolation] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if interpolation is not None:
            series, other = interpolation(series, other)

        self._validate_series_lengths(series)
        self._validate_series_lengths(other)

        n_jobs = kwargs.get("n_jobs", 1)

        with tqdm_joblib(
            tqdm(desc=self.name, total=len(series) * len(other), disable=not kwargs.get("verbose", False))
        ):
            distance_matrix = Parallel(n_jobs=n_jobs)(
                delayed(self._get_dist_fn(**kwargs))(series[i], other[j])  # only 1d metrics so far
                for i in range(len(series))
                for j in range(len(other))
            )
        return np.array(distance_matrix).reshape(len(series), len(other))

    def __call__(
        self, series_a: np.ndarray, series_b: np.ndarray, interpolation: Optional[Interpolation] = None, **kwargs: Any
    ) -> float:  # pairwise
        self._validate_series_lengths([series_a, series_b])
        if interpolation is not None:
            series = interpolation([series_a, series_b])
            series_a, series_b = series[0], series[1]
        return self._get_dist_fn(**kwargs)(series_a, series_b)

    def _validate_series_lengths(self, series: Union[np.ndarray, List[np.ndarray]]) -> None:
        assert (
            self in DistanceMetric.same_lengths_metrics() and np.array(series).ndim == 2
        ) or self in DistanceMetric.various_lengths_metrics(), (
            f"The metric {self.name} cannot handle time series with different lengths."
        )

    @staticmethod
    def various_lengths_metrics() -> List[DistanceMetric]:
        return [
            DistanceMetric.CROSS_CORRELATION,
            DistanceMetric.DTW,
            DistanceMetric.LONGEST_COMMON_SUBSEQUENCE,
            DistanceMetric.SHAPE_BASED_DISTANCE,
            DistanceMetric.MSM,
        ]

    @staticmethod
    def same_lengths_metrics() -> List[DistanceMetric]:
        return list(set(DistanceMetric.__members__.values()) - set(DistanceMetric.various_lengths_metrics()))

    def aggregate(
        self, series: Union[np.ndarray, List[np.ndarray]], method: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        if self == self.SHAPE_BASED_DISTANCE:
            if method == "sum":
                min_len = min([a_.shape[0] for a_ in series])
                return np.array(sum([a_[:min_len] for a_ in series]))
            elif method == "sum_max_len":
                max_len = max([a_.shape[0] for a_ in series])
                return np.array(sum([np.pad(a_, (0, max_len - a_.shape[0]), mode="constant") for a_ in series]))
            elif method == "mean":
                min_len = min([a_.shape[0] for a_ in series])
                return np.array(sum([a_[:min_len] for a_ in series]) / len(series))
            elif method == "median":
                distance_matrix = kwargs["distance_matrix"]
                median_idx = distance_matrix.sum(axis=0).argmin()
                return series[median_idx]
            elif method == "element_median":
                min_len = min([a_.shape[0] for a_ in series])
                return np.median([a_[:min_len] for a_ in series], axis=0)
            elif method == "element_median_max_len":
                max_len = max([a_.shape[0] for a_ in series])
                return np.median([np.pad(a_, (0, max_len - a_.shape[0]), mode="constant") for a_ in series], axis=0)
            elif method == "max":
                min_len = min([a_.shape[0] for a_ in series])
                return np.max([a_[:min_len] for a_ in series], axis=0)
            elif method == "median_median_len":
                median_len = int(np.median([a_.shape[0] for a_ in series]))
                return np.median(
                    [
                        a_[:median_len]
                        if a_.shape[0] >= median_len
                        else np.pad(a_, (0, median_len - a_.shape[0]), mode="constant")
                        for a_ in series
                    ],
                    axis=0,
                )
            elif method == "sum_median_len":
                median_len = int(np.median([a_.shape[0] for a_ in series]))
                return np.sum(
                    [
                        a_[:median_len]
                        if a_.shape[0] >= median_len
                        else np.pad(a_, (0, median_len - a_.shape[0]), mode="constant")
                        for a_ in series
                    ],
                    axis=0,
                )
            elif method == "concatenate":
                return np.concatenate(series)
            else:
                raise ValueError(f"Unknown method {method}")
        raise ValueError("This distance measure cannot be aggregated")


def estimate_combined_distance_variance(
    a: List[np.ndarray],
    b: List[np.ndarray],
    distance_matrix_a: np.ndarray,
    distance_matrix_b: np.ndarray,
    metric: DistanceMetric,
) -> float:
    var_a = np.var(distance_matrix_a)
    var_b = np.var(distance_matrix_b)
    medoid_a = a[np.argmin(distance_matrix_a.sum(axis=0))]
    medoid_b = b[np.argmin(distance_matrix_b.sum(axis=0))]
    distance_medoids = metric(medoid_a, medoid_b)
    full_distance_matrix = metric.matrix(a + b)

    estimated_variance = (
        var_a * (len(a) ** 2) + var_b * (len(b) ** 2) + ((distance_medoids / 2) * (len(a) * len(b)))
    ) / (len(a) ** 2 + len(b) ** 2 + len(a) * len(b))

    print(f"variance a: {var_a}")
    print(f"variance b: {var_b}")
    print(f"actual variance: {np.var(full_distance_matrix)}")
    print(f"estimated variance: {estimated_variance}")


if __name__ == "__main__":
    for _ in range(1):
        a = [np.random.randn(100) + 10 for _ in range(10)]
        b = [np.random.randn(100) * 3 for _ in range(10)]
        distance_matrix_a = DistanceMetric.SHAPE_BASED_DISTANCE.matrix(a)
        distance_matrix_b = DistanceMetric.SHAPE_BASED_DISTANCE.matrix(b)
        estimate_combined_distance_variance(
            a, b, distance_matrix_a, distance_matrix_b, DistanceMetric.SHAPE_BASED_DISTANCE
        )
