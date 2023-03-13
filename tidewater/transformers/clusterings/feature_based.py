from typing import Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pydantic.dataclasses import dataclass
from tslearn.utils import to_time_series_dataset
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from .base import DynSizeClustering


@dataclass
class FeatureBased(DynSizeClustering):
    n_clusters: int = 3
    max_iter: int = 50
    n_jobs: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=self.random_state)
        trans = self._transform_format(X)
        print(trans["value-0"])
        feats = extract_features(
            trans,
            n_jobs=self.n_jobs,
            column_id="id",
            column_sort="time",
            column_kind=None,
            column_value="value",
            default_fc_parameters=ComprehensiveFCParameters(),
        )
        feats = feats[
            [
                "value-0__variance_larger_than_standard_deviation",
                "value-0__has_duplicate_max",
                "value-0__has_duplicate_min",
                "value-0__has_duplicate",
                "value-0__sum_values",
                "value-0__abs_energy",
                "value-0__mean_abs_change",
                "value-0__mean_change",
                "value-0__mean_second_derivative_central",
                "value-0__median",
                "value-0__mean",
                "value-0__length",
                "value-0__standard_deviation",
                "value-0__variation_coefficient",
                "value-0__variance",
                "value-0__skewness",
                "value-0__kurtosis",
                "value-0__root_mean_square",
                "value-0__absolute_sum_of_changes",
                "value-0__longest_strike_below_mean",
                "value-0__longest_strike_above_mean",
                "value-0__count_above_mean",
                "value-0__count_below_mean",
                "value-0__last_location_of_maximum",
                "value-0__first_location_of_maximum",
                "value-0__last_location_of_minimum",
                "value-0__first_location_of_minimum",
                "value-0__percentage_of_reoccurring_values_to_all_values",
                "value-0__percentage_of_reoccurring_datapoints_to_all_datapoints",
                "value-0__sum_of_reoccurring_values",
                "value-0__sum_of_reoccurring_data_points",
                "value-0__ratio_value_number_to_time_series_length",
                "value-0__maximum",
                "value-0__minimum",
                "value-0__benford_correlation",
            ]
        ].dropna(axis=1)
        labels: np.ndarray = model.fit_predict(feats)
        return labels

    def _transform_format(self, X: List[np.ndarray]) -> dict:
        return {
            "value-0": pd.DataFrame(
                [{"id": j, "time": i, "value": x_i} for j, x in enumerate(X) for i, x_i in enumerate(x)]
            )
        }
