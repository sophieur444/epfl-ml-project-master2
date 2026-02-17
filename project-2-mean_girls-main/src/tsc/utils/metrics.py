# src/tsc/utils/metrics.py

import numpy as np


def accuracy(y_pred, y_true):
    """Compute accuracy of predictions."""
    return np.mean(y_true == y_pred)


def precision(y_pred, y_true):
    """Compute precision of predictions (positive class == 1)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_pred, y_true):
    """Compute recall of predictions (positive class == 1)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_pred, y_true):
    """Compute F1 score of predictions."""
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)
