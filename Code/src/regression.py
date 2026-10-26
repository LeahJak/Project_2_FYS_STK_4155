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
    Solves min ||y - X theta||^2 + lambda ||theta||^2."""
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ np.diag(s / (s**2 + n * lmbda) @ U.T @ y)
