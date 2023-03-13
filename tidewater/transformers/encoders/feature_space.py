from __future__ import annotations

from typing import Any, Dict, Optional, List, Union

from abc import abstractmethod
import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from dataclasses import field
from logging import warn
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.selection import select_features
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import rand_score, homogeneity_score, silhouette_score

from skopt import gp_minimize

from tidewater.datatypes.base import NumpyType
from tidewater.datatypes import Labels
from tidewater.transformers.interface import InputInterface


from .base import Encoder


@dataclass
class FeatureSpace(Encoder):
    n_jobs: int = 1
    verbose: bool = False
    default_fc_parameters: Optional[dict] = field(default_factory=lambda: ComprehensiveFCParameters())

    def __post_init__(self) -> None:
        super().__init__()

    def _get_important_features(self, data: List[np.ndarray]) -> List[str]:
        return [
            "value-0__variance_larger_than_standard_deviation",
            "value-0__has_duplicate_max",
            "value-0__has_duplicate_min",
            "value-0__sum_values",
            "value-0__mean_abs_change",
            "value-0__mean_second_derivative_central",
            "value-0__mean",
            "value-0__standard_deviation",
            "value-0__skewness",
            "value-0__kurtosis",
            "value-0__last_location_of_maximum",
            "value-0__first_location_of_maximum",
            "value-0__last_location_of_minimum",
            "value-0__first_location_of_minimum",
            "value-0__percentage_of_reoccurring_values_to_all_values",
            "value-0__ratio_value_number_to_time_series_length",
            "value-0__minimum",
            "value-0__benford_correlation",
            "value-0__time_reversal_asymmetry_statistic__lag_1",
            "value-0__symmetry_looking__r_0.05",
            "value-0__symmetry_looking__r_0.30000000000000004",
            "value-0__symmetry_looking__r_0.45",
            "value-0__symmetry_looking__r_0.5",
            "value-0__symmetry_looking__r_0.6000000000000001",
            "value-0__symmetry_looking__r_0.7000000000000001",
            "value-0__symmetry_looking__r_0.8",
            "value-0__symmetry_looking__r_0.9",
            "value-0__symmetry_looking__r_0.9500000000000001",
            "value-0__large_standard_deviation__r_0.05",
            "value-0__large_standard_deviation__r_0.15000000000000002",
            "value-0__large_standard_deviation__r_0.30000000000000004",
            "value-0__large_standard_deviation__r_0.35000000000000003",
            "value-0__large_standard_deviation__r_0.45",
            "value-0__large_standard_deviation__r_0.55",
            "value-0__large_standard_deviation__r_0.7000000000000001",
            "value-0__large_standard_deviation__r_0.8",
            "value-0__large_standard_deviation__r_0.8500000000000001",
            "value-0__large_standard_deviation__r_0.9",
            "value-0__quantile__q_0.4",
            "value-0__quantile__q_0.7",
            "value-0__quantile__q_0.8",
            "value-0__quantile__q_0.9",
            "value-0__autocorrelation__lag_1",
            "value-0__autocorrelation__lag_3",
            "value-0__autocorrelation__lag_4",
            "value-0__autocorrelation__lag_5",
            "value-0__autocorrelation__lag_6",
            "value-0__autocorrelation__lag_7",
            "value-0__autocorrelation__lag_8",
            'value-0__agg_autocorrelation__f_agg_"median"__maxlag_40',
            "value-0__partial_autocorrelation__lag_2",
            "value-0__partial_autocorrelation__lag_4",
            "value-0__number_cwt_peaks__n_1",
            "value-0__number_cwt_peaks__n_5",
            "value-0__number_peaks__n_3",
            "value-0__number_peaks__n_5",
            "value-0__number_peaks__n_50",
            "value-0__binned_entropy__max_bins_10",
            "value-0__index_mass_quantile__q_0.1",
            "value-0__index_mass_quantile__q_0.2",
            "value-0__index_mass_quantile__q_0.3",
            "value-0__index_mass_quantile__q_0.4",
            "value-0__index_mass_quantile__q_0.7",
            "value-0__cwt_coefficients__coeff_0__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)",
            "value-0__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)",
            "value-0__ar_coefficient__coeff_2__k_10",
            "value-0__ar_coefficient__coeff_4__k_10",
            "value-0__ar_coefficient__coeff_5__k_10",
            "value-0__ar_coefficient__coeff_10__k_10",
            'value-0__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
            'value-0__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
            'value-0__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
            'value-0__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
            'value-0__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
            'value-0__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
            'value-0__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
            'value-0__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
            'value-0__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
            'value-0__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8',
            'value-0__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
            'value-0__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
            'value-0__fft_coefficient__attr_"real"__coeff_2',
            'value-0__fft_coefficient__attr_"real"__coeff_3',
            'value-0__fft_coefficient__attr_"real"__coeff_4',
            'value-0__fft_coefficient__attr_"real"__coeff_5',
            'value-0__fft_coefficient__attr_"imag"__coeff_0',
            'value-0__fft_coefficient__attr_"imag"__coeff_2',
            'value-0__fft_coefficient__attr_"imag"__coeff_4',
            'value-0__fft_coefficient__attr_"imag"__coeff_6',
            'value-0__fft_coefficient__attr_"abs"__coeff_0',
            'value-0__fft_coefficient__attr_"abs"__coeff_4',
            'value-0__fft_coefficient__attr_"abs"__coeff_5',
            'value-0__fft_coefficient__attr_"angle"__coeff_0',
            'value-0__fft_coefficient__attr_"angle"__coeff_1',
            'value-0__fft_coefficient__attr_"angle"__coeff_2',
            'value-0__fft_coefficient__attr_"angle"__coeff_5',
            'value-0__fft_coefficient__attr_"angle"__coeff_7',
            'value-0__fft_aggregated__aggtype_"kurtosis"',
            "value-0__range_count__max_1__min_-1",
            "value-0__range_count__max_1000000000000.0__min_0",
            "value-0__approximate_entropy__m_2__r_0.1",
            "value-0__approximate_entropy__m_2__r_0.5",
            "value-0__approximate_entropy__m_2__r_0.7",
            "value-0__approximate_entropy__m_2__r_0.9",
            'value-0__linear_trend__attr_"pvalue"',
            'value-0__linear_trend__attr_"rvalue"',
            'value-0__linear_trend__attr_"intercept"',
            'value-0__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
            'value-0__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
            'value-0__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
            'value-0__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
            'value-0__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
            'value-0__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
            'value-0__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
            'value-0__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
            'value-0__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
            'value-0__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
            "value-0__number_crossing_m__m_0",
            "value-0__number_crossing_m__m_1",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_1",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_2",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_5",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_6",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_7",
            "value-0__energy_ratio_by_chunks__num_segments_10__segment_focus_9",
            "value-0__ratio_beyond_r_sigma__r_0.5",
            "value-0__ratio_beyond_r_sigma__r_1",
            "value-0__ratio_beyond_r_sigma__r_1.5",
            "value-0__ratio_beyond_r_sigma__r_2",
            "value-0__ratio_beyond_r_sigma__r_10",
            "value-0__count_below__t_0",
            "value-0__fourier_entropy__bins_2",
            "value-0__fourier_entropy__bins_3",
            "value-0__fourier_entropy__bins_5",
            "value-0__fourier_entropy__bins_100",
            "value-0__permutation_entropy__dimension_4__tau_1",
            "value-0__permutation_entropy__dimension_5__tau_1",
            "value-0__permutation_entropy__dimension_6__tau_1",
        ]

    def _extract_features(self, data: List[np.ndarray]) -> pd.DataFrame:
        trans = self._transform_format(data)
        feats = extract_features(
            trans,
            n_jobs=self.n_jobs,
            column_id="id",
            column_sort="time",
            column_kind=None,
            column_value="value",
            default_fc_parameters=self.default_fc_parameters,
        ).dropna(axis=1)
        feats = feats[self._get_important_features(data)]
        post_processed = StandardScaler().fit_transform(feats)
        n_features = post_processed.shape[1]
        post_processed = VarianceThreshold(threshold=0.1).fit_transform(post_processed)
        if self.verbose and n_features != post_processed.shape[1]:
            print(f"Reduced features from {n_features} to {post_processed.shape[1]}")
        return pd.DataFrame(post_processed, columns=list(range(post_processed.shape[1])))

    def _encode(self, data: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        feats = self._extract_features(data)
        return feats.values

    def _transform_format(self, X: List[np.ndarray]) -> dict:
        return {
            "value-0": pd.DataFrame(
                [{"id": j, "time": i, "value": x_i} for j, x in enumerate(X) for i, x_i in enumerate(x)]
            )
        }


@dataclass
class FeatureSpaceOptimizeBase(FeatureSpace):
    @abstractmethod
    def _optimize(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        ...

    def _extract_features(self, data: List[np.ndarray]) -> pd.DataFrame:
        feats = super()._extract_features(data)
        feats = self._optimize(feats)
        return feats


@dataclass
class FeatureSpaceOptimizeUnsupervised(FeatureSpaceOptimizeBase):
    clusterer: Any = None
    n_calls: int = 30
    n_features: int = 10

    def _optimize(self, X: pd.DataFrame) -> pd.DataFrame:
        def f(x):
            # features = X.iloc[:, [bool(i) for i in x]].values
            print(x)
            features = X.iloc[:, x].values
            print(f"features: {features.shape[1]} of {X.shape[1]}")
            predicted = self.clusterer.fit_predict(features.tolist())
            score = silhouette_score(features, predicted)
            print(score)
            return -score

        res = gp_minimize(f, [(0, len(X.columns) - 1) for _ in range(self.n_features)], n_calls=self.n_calls)

        print(-res.fun, res.x)
        return X.iloc[:, res.x]


@dataclass
class FeatureSpaceOptimizeSupervised(FeatureSpaceOptimizeBase):
    clusterer: Any = None
    n_calls: int = 30

    def _optimize(self, X: pd.DataFrame) -> pd.DataFrame:
        labels = self.get_input_value("labels")[0]
        assert labels is not None and isinstance(
            labels, Labels
        ), f"{self.__class__.__name__} has not received all its inputs yet."
        l = labels.ndarray

        def f(x):
            chosen = [bool(i) for i in x]
            print(f"{X.iloc[:, chosen].values.shape} of {X.shape}")
            predicted = self.clusterer.fit_predict(X.iloc[:, chosen].values.tolist())
            score = homogeneity_score(l, predicted)
            print(score)
            simplicity_score = np.sum(chosen) / len(chosen)
            return -score + simplicity_score

        res = gp_minimize(f, [(0, 1) for x in X.columns], n_calls=self.n_calls)

        print(-res.fun, [bool(i) for i in res.x])
        return X.iloc[:, [bool(i) for i in res.x]]

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[NumpyType], labels=Labels)


class TopKVarianceThreshold:
    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray] = None) -> TopKVarianceThreshold:
        variances = np.var(X, axis=0)
        self.sorted_idx = np.argsort(variances)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        slicing = self.sorted_idx[-self.k :]
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, slicing]
        return X[:, slicing]

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
