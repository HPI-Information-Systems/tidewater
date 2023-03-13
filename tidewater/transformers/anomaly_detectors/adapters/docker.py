from __future__ import annotations

from typing import Type, Any, Optional

import numpy as np

from ..base import AnomalyDetector
from ...adapters import UnsupervisedDockerAdapter
from ....datatypes import TimeSeries, Scores
from ....datatypes.model import Model


class UnsupervisedDockerAnomalyDetector(UnsupervisedDockerAdapter, AnomalyDetector):
    """
    A base DockerAdapter for all dockerized anomaly detection algorithms.
    """

    @property
    def _numpy_input_type(self) -> Type[TimeSeries]:
        return TimeSeries

    @property
    def _numpy_return_type(self) -> Type[Scores]:
        return Scores
