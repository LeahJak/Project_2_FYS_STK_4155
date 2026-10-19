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

