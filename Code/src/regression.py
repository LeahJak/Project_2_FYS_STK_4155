# file for project 1 funstions 
# OlS , lasso 

import numpy as np

# The design matrix is now made using the function design_matrix in data.py 
# The train test split is made using make_runge_split 
# Standardising is done by scale_data 
# The only things lacking to recreate the data from project 1 is the ols closed form and ridge coefficients. 


def ols_fit(X, y):
    """Closed-form OLS coefficients (project 1, part a)."""
    return np.linalg.pinv(X) @ y


def ridge_fit(X, y, lmbda):
    """Ridge coefficients via the SVD (project 1, part b).
    Solves min (1/n)||y - X theta||^2 + lambda ||theta||^2."""
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ np.diag(s / (s**2 + n * lmbda)) @ U.T @ y

    
def lasso_fit(X, y, lam, gamma = None, n_iter = 2000, tol = 1e-10):
    """Lasso coefficients with plain gradient descent
    Minimises (1/n)||y - X theta||^2 + lam ||theta||_1, using the subgradient lam*sign(theta)"""

    n, p = X.shape
    y = np.asarray(y).reshape(n, -1)

    if gamma is None:
        gamma = n / (2 * np.linalg.norm(X, 2)**2)  
    theta = np.zeros((p, y.shape[1]))

    for _ in range(n_iter):
        grad = (2.0 / n) * X.T @ (X @ theta - y) + lam * np.sign(theta)
        theta_new = theta - gamma * grad
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


