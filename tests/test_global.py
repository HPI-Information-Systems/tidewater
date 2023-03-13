import tempfile
import unittest
import os
import pytest
from pathlib import Path

from tidewater.pipeline import Pipeline
from tidewater.pipeline.base import EdgeAttributes
from tidewater.transformers.data_handling.anomaly_slicer import AnomalySlicer
from tidewater.transformers.data_handling.loader import CSVTimeSeriesLoader
from tidewater.transformers.anomaly_detectors.kmeans import KMeans as KMeansAD
from tidewater.transformers.clusterings.kmeans import KMeans as KMeansCluster
from tidewater.transformers.data_handling.scaler import MinMaxScaler
from tidewater.transformers.data_handling.thresholder import Thresholder
from tidewater.transformers.data_handling.writer import Writer


@pytest.mark.skip
class TestGlobal(unittest.TestCase):
    def test_build_pipeline(self):
        p = Path("./results")
        p.mkdir(exist_ok=True)
        pipeline = Pipeline(results_path=p)
        dataloader = pipeline.add_transformer(
            CSVTimeSeriesLoader(Path(os.getcwd()) / "data/taxi.csv", index_col=True, label_col=2)
        )
        anomaly_detector = pipeline.add_transformer(KMeansAD())
        pipeline.add_connection(dataloader, anomaly_detector, EdgeAttributes("data", "data"))

        train_scaler = pipeline.add_transformer(MinMaxScaler.Training())
        pipeline.add_connection(anomaly_detector, train_scaler, EdgeAttributes("data", "data"))

        transform_scaler = pipeline.add_transformer(MinMaxScaler())
        pipeline.add_connection(anomaly_detector, transform_scaler, EdgeAttributes("data", "data"))
        pipeline.add_connection(train_scaler, transform_scaler, EdgeAttributes("model", "model"))

        thresholder = pipeline.add_transformer(Thresholder(0.7))
        pipeline.add_connection(transform_scaler, thresholder, EdgeAttributes("data", "data"))

        anomaly_slicer = pipeline.add_transformer(AnomalySlicer(min_len=2))
        pipeline.add_connection(dataloader, anomaly_slicer, EdgeAttributes("data", "timeseries"))
        pipeline.add_connection(thresholder, anomaly_slicer, EdgeAttributes("data", "labels"))

        clustering = pipeline.add_transformer(KMeansCluster(n_clusters=2))
        pipeline.add_connection(anomaly_slicer, clustering, EdgeAttributes("data", "data"))

        f = tempfile.NamedTemporaryFile()
        writer = pipeline.add_transformer(Writer(Path(f.name)))
        pipeline.add_connection(clustering, writer, EdgeAttributes("data", "data"))

        pipeline.execute()
