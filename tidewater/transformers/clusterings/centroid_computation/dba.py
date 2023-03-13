from typing import List

import numpy as np
from tslearn.barycenters import softdtw_barycenter

from .base import CentroidComputation


class DBA(CentroidComputation):
    """
    This Transformer performs the Dynamic Time Warping Barycenter Averaging.
    """

    def _calculate_centroid(self, timeseries: List[np.ndarray], labels: np.ndarray) -> List[np.ndarray]:
        unique_classes = np.unique(labels)
        unique_classes.sort()
        centroids = []
        for cl in unique_classes:
            idx = np.where(labels == cl)[0]
            centroid = softdtw_barycenter([timeseries[i] for i in idx])
            centroids.append(centroid)
        return centroids
