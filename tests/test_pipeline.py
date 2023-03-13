import unittest
from typing import Any, Optional

import numpy as np

from tidewater.datatypes import TimeSeries
from tidewater.datatypes.base import BaseDataType
from tidewater.datatypes.model import Model
from tidewater.pipeline import Pipeline
from tidewater.pipeline.base import EdgeAttributes
from tidewater.transformers.anomaly_detectors import KMeans
from tidewater.transformers.base import Transformer
from tidewater.transformers.interface import InputInterface, OutputInterface


class DummyDataType(BaseDataType):
    pass


class DummyTransformer1(Transformer):
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Model:
        pass

    def execute(self, **kwargs):
        input_value = self.get_input_value("data")[0]
        assert input_value is not None, "Test Error"
        self.set_output_value(data=input_value)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=TimeSeries)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=TimeSeries)


class DummyTransformer2(Transformer):
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Model:
        pass

    def execute(self, **kwargs):
        self.set_output_value(result=DummyDataType())

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data1=TimeSeries, data2=TimeSeries)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(result=DummyDataType)


class TestPipeline(unittest.TestCase):
    def test_pipeline_starting_transformers(self):
        pipeline = Pipeline()
        t0 = pipeline.add_transformer(KMeans())
        t1 = pipeline.add_transformer(KMeans())
        pipeline.add_connection(t0, t1, edge_attr=EdgeAttributes("test0", "test1"))
        starters = pipeline._get_starting_transformers()
        self.assertEqual(starters, [t0])

    def test_pipeline_next_transformers(self):
        pipeline = Pipeline()
        t0 = pipeline.add_transformer(KMeans())
        t1 = pipeline.add_transformer(KMeans())
        t2 = pipeline.add_transformer(KMeans())
        pipeline.add_connection(t0, t1, EdgeAttributes("test0", "test1"))
        pipeline.add_connection(t0, t2, EdgeAttributes("test0", "test1"))
        next_transformers = pipeline._get_next_transformers(t0)
        try:
            self.assertListEqual(next_transformers, [t1, t2])
        except:
            self.assertListEqual(next_transformers, [t2, t1])
        self.assertListEqual(pipeline._get_next_transformers(t1), [])

    def test_pipeline_execution_order_forking(self):
        pipeline = Pipeline()
        t0 = pipeline.add_transformer(KMeans())
        t1 = pipeline.add_transformer(KMeans())
        t2 = pipeline.add_transformer(KMeans())
        pipeline.add_connection(t0, t1, EdgeAttributes("test0", "test1"))
        pipeline.add_connection(t0, t2, EdgeAttributes("test0", "test1"))
        order = pipeline._build_execution_order()
        try:
            self.assertListEqual(order, [t0, t2, t1])
        except:
            self.assertListEqual(order, [t0, t1, t2])

    def test_pipeline_execution_order_merging(self):
        pipeline = Pipeline()
        t0 = pipeline.add_transformer(KMeans())
        t1 = pipeline.add_transformer(KMeans())
        t2 = pipeline.add_transformer(KMeans())
        pipeline.add_connection(t1, t0, EdgeAttributes("test0", "test1"))
        pipeline.add_connection(t2, t0, EdgeAttributes("test0", "test1"))
        order = pipeline._build_execution_order()
        self.assertListEqual(order, [t1, t2, t0])

    def test_pipeline_execution(self):
        pipeline = Pipeline()
        t0 = pipeline.add_transformer(DummyTransformer1())
        t1 = pipeline.add_transformer(DummyTransformer1())
        t2 = pipeline.add_transformer(DummyTransformer2())
        pipeline.add_connection(t0, t2, EdgeAttributes("data", "data1"))
        pipeline.add_connection(t1, t2, EdgeAttributes("data", "data2"))

        ts = TimeSeries(ndarray=np.ndarray([1, 2, 3, 4, 5]))
        pipeline.get_transformer(t0).set_input_value(data=ts)
        pipeline.get_transformer(t1).set_input_value(data=ts)
        pipeline.execute()
        self.assertIsNotNone(pipeline.get_transformer(t2).get_output_value("result"))

    def test_pipeline_validation_disconnected(self):
        with self.assertRaises(ValueError):
            pipeline = Pipeline()
            t0 = pipeline.add_transformer(DummyTransformer1())
            t1 = pipeline.add_transformer(DummyTransformer1())
            t2 = pipeline.add_transformer(DummyTransformer2())
            pipeline._validate_graph()

    def test_pipeline_validation_cycle(self):
        with self.assertRaises(ValueError):
            pipeline = Pipeline()
            t0 = pipeline.add_transformer(DummyTransformer1())
            t1 = pipeline.add_transformer(DummyTransformer1())
            t2 = pipeline.add_transformer(DummyTransformer2())

            pipeline.add_connection(t0, t1, EdgeAttributes("data", "data"))
            pipeline.add_connection(t1, t2, EdgeAttributes("data", "data"))
            pipeline.add_connection(t2, t0, EdgeAttributes("data", "data"))

            pipeline._validate_graph()
