from typing import Any, Optional, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


from tidewater.datatypes import TimeSeries, Labels
from tidewater.datatypes.model import Model
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface


class Plotter(Transformer):
    """
    This Transformer plots TimeSeries and Centroids.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = 0

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def _plot_multiple_timeseries(self, tss: List[Tuple[TimeSeries, Labels]], fig: go.Figure, title: str) -> None:
        for ts, anom in tss:
            color = "red" if np.max(anom.ndarray) == 1 else "grey"
            dash = "dash" if np.max(anom.ndarray) == 1 else "solid"
            fig.add_trace(
                go.Scatter(
                    y=ts.to_2d()[:, self.dim],
                    marker=dict(color=color),
                    line=dict(dash=dash),
                    legendgroup=title,
                    showlegend=False,
                    opacity=0.3,
                )
            )

    def _plot_single(self, ts: TimeSeries, fig: go.Figure, title: str, group: str) -> None:
        fig.add_trace(go.Scatter(y=ts.to_2d()[:, self.dim], name=title, legendgroup=group))

    def execute(self, **kwargs: Any) -> None:
        data, labels, anomalies = self.get_input_value("data", "labels", "anomalies")
        assert (data is not None) and (labels is not None) and (anomalies is not None), "Input data is not filled yet."
        assert isinstance(data, List) and isinstance(data[0], TimeSeries), "Wrong data types"
        assert isinstance(labels, Labels), "Wrong data types"
        assert isinstance(anomalies, List) and isinstance(anomalies[0], Labels), "Wrong data types"

        unique_classes = np.unique(labels.ndarray)
        unique_classes.sort()
        fig = go.Figure()
        for cl in unique_classes:
            title = f"Cluster #{cl}"
            tss = [(ts, a) for ts, l, a in zip(data, labels.ndarray, anomalies) if l == cl]
            self._plot_multiple_timeseries(tss, fig, title)
            fig.add_trace(go.Scatter(y=np.zeros(10), name=title, legendgroup=title))  # phantom to get groups in legend
        fig.show()

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels, anomalies=List[Labels])  # type: ignore


class PlotterCentroids(Transformer):
    """
    This Transformer plots TimeSeries and Centroids.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = 0

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def _plot_multiple_timeseries(self, tss: List[Tuple[TimeSeries, Labels]], fig: go.Figure, title: str) -> None:
        for ts, anom in tss:
            color = "red" if np.max(anom.ndarray) == 1 else "grey"
            dash = "dash" if np.max(anom.ndarray) == 1 else "solid"
            fig.add_trace(
                go.Scatter(
                    y=ts.ndarray[:, self.dim],
                    marker=dict(color=color),
                    line=dict(dash=dash),
                    legendgroup=title,
                    showlegend=False,
                    opacity=0.3,
                )
            )

    def _plot_single(self, ts: TimeSeries, fig: go.Figure, title: str, group: str) -> None:
        fig.add_trace(go.Scatter(y=ts.to_2d()[:, self.dim], name=title, legendgroup=group))

    def execute(self, **kwargs: Any) -> None:
        data, labels, centroids, anomalies = self.get_input_value("data", "labels", "centroids", "anomalies")
        assert (
            (data is not None) and (labels is not None) and (centroids is not None) and (anomalies is not None)
        ), "Input data is not filled yet."
        assert isinstance(data, List) and isinstance(data[0], TimeSeries), "Wrong data types"
        assert isinstance(labels, Labels), "Wrong data types"
        assert isinstance(centroids, List) and isinstance(centroids[0], TimeSeries), "Wrong data types"
        assert isinstance(anomalies, List) and isinstance(anomalies[0], Labels), "Wrong data types"

        unique_classes = np.unique(labels.ndarray)
        unique_classes.sort()
        fig = go.Figure()
        for cl in unique_classes:
            tss = [(ts, a) for ts, l, a in zip(data, labels.ndarray, anomalies) if l == cl]
            self._plot_multiple_timeseries(tss, fig, f"Cluster #{cl}")
            self._plot_single(centroids[cl], fig, f"Centroid #{cl}", f"Cluster #{cl}")
        fig.show()

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels, centroids=List[TimeSeries], anomalies=List[Labels])  # type: ignore


class SpacePlotter(Plotter):
    def execute(self, **kwargs: Any) -> None:
        data, labels, anomalies, distance_matrix = self.get_input_value(
            "data", "labels", "anomalies", "distance_matrix"
        )
        assert (data is not None) and (labels is not None) and (anomalies is not None), "Input data is not filled yet."
        assert isinstance(data, List) and isinstance(data[0], TimeSeries), "Wrong data types"
        assert isinstance(labels, Labels), "Wrong data types"
        assert isinstance(anomalies, List) and isinstance(anomalies[0], Labels), "Wrong data types"
        assert isinstance(distance_matrix, List) and isinstance(distance_matrix[0], TimeSeries), "Wrong data types"

        pca = PCA(n_components=2)
        distance_matrix_np = np.array([d.ndarray for d in distance_matrix])
        space = pca.fit_transform(distance_matrix_np)

        anomaly_idx = [i for i, anom in enumerate(anomalies) if np.max(anom.ndarray) == 1]
        normal_idx = [i for i, anom in enumerate(anomalies) if np.max(anom.ndarray) != 1]
        shapelet_idx = np.where(distance_matrix_np.min(axis=1) == 0)[0]

        fig = go.Figure()
        fig.add_scatter(
            x=space[anomaly_idx, 0], y=space[anomaly_idx, 1], fillcolor="red", mode="markers", name="anomalies"
        )
        fig.add_scatter(x=space[normal_idx, 0], y=space[normal_idx, 1], fillcolor="grey", mode="markers", name="normal")
        fig.add_scatter(
            x=space[shapelet_idx, 0], y=space[shapelet_idx, 1], fillcolor="blue", mode="markers", name="shapelets"
        )

        fig.show()

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=List[TimeSeries], labels=Labels, anomalies=List[Labels], distance_matrix=List[TimeSeries])  # type: ignore
