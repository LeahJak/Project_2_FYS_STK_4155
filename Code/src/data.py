# Data generation and loading: Runge function data and MNIST.
# Everyone loads data from here so it is identical across the group.

import numpy as np 
from sklearn.model_selection import train_test_split 

seed = 2026 

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.column_stack([x**p for p in range(start, degree + 1)])

def make_runge(n=100, sigma=0.1, seed=seed):
    """ For the network the input is the raw x (one feature), samples in rows"""
    rng = np.random.default_rng(seed)      # fix the seed for reproducible results
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y = runge(x) + rng.normal(0.0, sigma, n)
    return x.reshape(-1, 1), y.reshape(-1,1)

def make_runge_split(n=100, sigma=0.1, test_size=0.2, seed=seed):
    X, y = make_runge(n, sigma, seed)
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def scale_data(X_train, X_test):
    """
    Standardize features using the training data.
    Parameters:
    X_train : array-like. Training features.
    X_test : array-like. Test features.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std 


def center_data(y_train, y_test):
    """Center targets using the training-target mean."""
    mean = np.mean(y_train)
    y_train_centered = y_train - mean
    y_test_centered = y_test - mean
    return y_train_centered, y_test_centered, mean


def load_mnist(path="mnist.npz"):
    """Loading full MNIST data: images flattened to 784 values and scaled to [0, 1], labels as digits 0-9."""
    with np.load(path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0
    return x_train, y_train, x_test, y_test


def one_hot(y, n_classes=10):
    """Digits 0-9 (labels) -> one-hot rows, e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]."""
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

def make_mnist_split(path="mnist.npz", val_size=10000, seed=seed):
    """MNIST with a validation set split off from the training data. the test set is left untouched."""
    x_train, y_train, x_test, y_test = load_mnist(path)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)
    return x_tr, x_val, x_test, y_tr, y_val, y_test

def make_runge_split_3(n=100, sigma=0.1, test_size=0.2, val_size=0.2, seed = seed):
    """Runge data split into training, validation and test sets"""
    x_train, x_test, y_train, y_test = make_runge_split(n, sigma, test_size, seed)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)
    return x_tr, x_val, x_test, y_tr, y_val, y_test  