import unittest

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from tidewater.datatypes import Labels
from tidewater.transformers.metrics.base import Precision, Recall, F1


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.true_labels = Labels(ndarray=np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0]))
        self.pred_labels = Labels(ndarray=np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))

    def test_correct_precision(self):
        metric = Precision()
        metric.set_input_value(true_labels=self.true_labels, pred_labels=self.pred_labels)
        metric.execute()
        self.assertEqual(metric.result, precision_score(self.true_labels.ndarray, self.pred_labels.ndarray))

    def test_correct_recall(self):
        metric = Recall()
        metric.set_input_value(true_labels=self.true_labels, pred_labels=self.pred_labels)
        metric.execute()
        self.assertEqual(metric.result, recall_score(self.true_labels.ndarray, self.pred_labels.ndarray))

    def test_correct_f1(self):
        metric = F1()
        metric.set_input_value(true_labels=self.true_labels, pred_labels=self.pred_labels)
        metric.execute()
        self.assertEqual(metric.result, f1_score(self.true_labels.ndarray, self.pred_labels.ndarray))
