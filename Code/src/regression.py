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

    

def lasso_fit(X, y, lam, gamma = None, n_iter = 5000000, tol = 1e-10, beta1=0.9, beta2=0.999, eps=1e-8):
    """Lasso coefficients with plain optimization
    Minimises (1/n)||y - X theta||^2 + lam ||theta||_1, using the subgradient lam*sign(theta)"""

    n, p  = X.shape
    y = np.asarray(y).reshape(n, -1)

    if gamma is None:
        gamma = n/ (2 * np.linalg.norm(X, 2)**2) # 1/lam_max of the MSE Hessian

    theta = np.zeros((p, y.shape[1]))
    m = np.zeros_like(theta) # moving average of grads
    r = np.zeros_like(theta) # moving average of squared grads

    for i in range(1, n_iter+1): # bias corrections
        grad = (2.0 / n) * X.T @ (X @ theta - y) + lam * np.sign(theta)

        # adam optimizer
        m = beta1 * m + (1.0 - beta1) * grad
        r = beta2 * r + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1**i)
        r_hat = r / (1.0 - beta2**i)
        theta_new = theta - (gamma * m_hat / (np.sqrt(r_hat) + eps))

        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta

