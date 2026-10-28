# Evaluation metrics: MSE, R2, accuracy, confusion matrix.
# Owner: Person 3

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def accuracy(predictions, targets):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

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
    "An accuracy p̂ on n test images is a binomial proportion with standard error: "
    return np.sqrt(p * (1 - p) / n)

def r2(predict, target):
    """R2 score: 1 - sum_i(y_i - y_tudl_i)^2 / sum_i(y_i - mean(y))^2"""
    predict = np.asarray(predict).ravel()
    target = np.asarray(target).ravel()
    return r2_score(target, predict)