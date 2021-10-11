#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-10-11
"""

import numpy as np
from sklearn import metrics


class ClassificationMeter(object):

    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._pred_list = []
        self._true_list = []

    def update(self,
               y_pred: np.ndarray,
               y_true: np.ndarray):
        assert len(y_pred.shape) >= 1
        assert len(y_true.shape) >= 1
        self._pred_list.extend(y_pred)
        self._true_list.extend(y_true)

    def accuracy_score(self):
        y_true = np.array(self._true_list)
        y_pred = np.array(self._pred_list)
        return metrics.accuracy_score(y_true, y_pred)
