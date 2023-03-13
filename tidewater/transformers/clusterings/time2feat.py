from typing import Any, Optional, List

import numpy as np
from pydantic.dataclasses import dataclass

from .base import DynSizeClustering

from t2f.extractor import feature_extraction
from t2f.importance import feature_selection
from t2f.clustering import ClusterWrapper


@dataclass
class Time2Feat(DynSizeClustering):
    n_clusters: int = 3
    transform_type: str = "std"
    model_type: str = "KMeans"

    def __post_init__(self) -> None:
        super().__init__()

    def _cluster_transform(self, X: List[np.ndarray], **kwargs: Any) -> np.ndarray:
        # Feature extraction

        X = [x.reshape(-1, 1) for x in X]

        df_feats = feature_extraction(X, batch_size=100, p=1)

        # Feature selection
        context = {"model_type": self.model_type, "transform_type": self.transform_type}
        top_feats = feature_selection(df_feats, context=context)
        df_feats = df_feats[top_feats]

        # Clustering
        model = ClusterWrapper(
            n_clusters=self.n_clusters, model_type=self.model_type, transform_type=self.transform_type
        )
        y_pred = model.fit_predict(df_feats)
        return y_pred
