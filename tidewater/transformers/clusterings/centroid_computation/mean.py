from typing import List

import numpy as np

from tidewater.transformers.clusterings.centroid_computation.base import CentroidComputation


class Mean(CentroidComputation):
    def _calculate_centroid(self, timeseries: List[np.ndarray], labels: np.ndarray) -> List[np.ndarray]:
        unique_classes = np.unique(labels)
        unique_classes.sort()
        ts = np.stack(timeseries)
        centroids = []
        for cl in unique_classes:
            idx = np.where(labels == cl)[0]
            centroid = ts[idx].mean(axis=0)
            centroids.append(centroid)
        return centroids
