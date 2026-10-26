# Evaluation metrics: MSE, R2, accuracy, confusion matrix.
# Owner: Person 3

import numpy as np


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