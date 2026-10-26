# file for project 1 funstions 
# OlS , lasso 

import numpy as np

def ols_fit(X, y):
    """Closed-form OLS coefficients (project 1, part a)"""
    "Solves min (1/n) ||y - X theta||^2 + lambda ||theta||^2"
    return np.linalg.pinv(X) @ y

def ridge_fit(X, y, lam):
    """Ridge coefficients via the SVD (project 1, part b)"""
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False) # Finding singular value decomposition, using numpy module
    return Vt.T @ (s / (s**2 + n * lam) * (U.T @ y)) # returning estimated Ridge coefficients/theta values


def lasso_fit(X, y, lam, gamma = None, n_iter = 2000, tol = 1e10):
    """Lasso coefficients with plain gradient descent (project 1)"""

    n, p = X.shape()
    y = np.asarray(y).reshape(n, -1)

    if gamma is None:
        gamma = n / (2 * np.linalg.norm(X, 2)**2)
        theta = np.zero((p, y.skape[1]))

    for _ in range(n_iter):
        grad = (2.0 / n) * X.T @ (X @ theta - y) + lam * np.sign(theta)
        theta_new = theta - gamma * grad
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


