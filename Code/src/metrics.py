# Evaluation metrics: MSE, R2, accuracy, confusion matrix.
# Owner: Person 3

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def accuracy(y_true, y_pred):
    """Fraction of correct predictions: the mean of the indicator I(t_i = y_i)."""
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """C[i, j] = number of samples of true class i predicted as class j."""
    C = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(C, (np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)), 1)
    return C


def per_class_from_confusion(C):
    """Precision and recall of every class from the K x K matrix: each class against the rest."""
    tp = np.diag(C).astype(float)
    precision = tp / np.maximum(C.sum(axis=0), 1)
    recall = tp / np.maximum(C.sum(axis=1), 1)
    return precision, recall

def standard_error(p, n):
    """Standard error of a proportion p (e.g. accuracy) measured on n samples, Eq. (2.24)."""
    return np.sqrt(p * (1 - p) / n)

def mse(predict, target):
    """mean squared error: (1/n) * sum_i (y_i - y_tudl_i)^2"""
    predict = np.asarray(predict).ravel()
    target = np.asarray(target).ravel()
    return mean_squared_error(target, predict)

def r2(predict, target):
    """R2 score: 1 - sum_i(y_i - y_tudl_i)^2 / sum_i(y_i - mean(y))^2"""
    predict = np.asarray(predict).ravel()
    target = np.asarray(target).ravel()
    return r2_score(target, predict)