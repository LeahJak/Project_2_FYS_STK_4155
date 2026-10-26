<<<<<<< HEAD
# FYS-STK3155/4155 Project 2: Feed-forward neural networks

Group members: Nora Arnesen, Viktoria Krogen, Natalie Kristiansen, Leah Jakobsen 


## Repository layout

```
Project2/
├── Code/
│   ├── src/            Shared code. Notebooks import from here.
│   │   ├── functions.py    costs, activations, derivatives
│   │   ├── network.py      FFNN + backprop
│   │   ├── optimizers.py   GD, SGD, RMSprop, Adam
│   │   ├── data.py         Runge data, MNIST loading
│   │   ├── metrics.py      MSE, R2, accuracy, confusion matrix
│   │   ├── plotting.py     shared plot style and helpers
│   │   └── regression.py   code snippets from project 1, OLS, Ridge, Lasso
│   ├── notebooks/      One notebook per part / per person
│   ├── tests/          pytest tests
│   ├── pyproject.toml
│   └── requirements.txt
├── figures/            All figures used in the report
└── Report/             PDF of the final report
```

## Setup (each person, once)

```bash
git clone <repo-url>
cd Project2/Code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `from src... import ...` work in notebooks
cd ..
nbstripout --install               # strips notebook outputs on commit
nbdime config-git --enable         # notebook-aware git diff/merge
```

## Reproducing the results

Open each notebook in `Code/notebooks/` and run **Kernel → Restart & Run All**.
Figures are saved to `figures/`.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Use of LLMs

_Describe how AI tools were used, as required by the project text._

## Part a - The analytical warm up 

### The mean squared error 
The MSE measures the average squared residual between the true targets
$y_i$ and the predictions $\tilde{y}_i = \mathbf{X}_{i,*}\boldsymbol{\theta}$.

$$
\mathrm{MSE}(\boldsymbol{\theta}) =
\frac{1}{n}\sum_{i=1}^{n}(y_i-\tilde{y}_i)^2
$$

The gradient with respect to the parameter vector is

$$
\nabla_{\boldsymbol{\theta}}\mathrm{MSE} =
-\frac{2}{n}\mathbf{X}^T
(\mathbf{y}-\mathbf{X}\boldsymbol{\theta})
$$

n is the number of observations, (\mathbf{X}_{i,*}) is the (i)-th row of the design matrix, and (\boldsymbol{\theta}) is the parameter vector.

### L1 regularization (Lasso)

Lasso regression adds an L1-norm penalty to the MSE, penalising the absolute values of the model parameters:

$$
J(\boldsymbol{\theta}) =
\mathrm{MSE}(\boldsymbol{\theta}) +
\lambda \|\boldsymbol{\theta}\|_1
$$

where

$$
\|\boldsymbol{\theta}\|_1 =
\sum_{j=0}^{p-1}|\theta_j|.
$$

The gradient, for nonzero penalized coefficients, is

$$
\nabla_{\boldsymbol{\theta}}J =
-\frac{2}{n}\mathbf{X}^T
(\mathbf{y}-\mathbf{X}\boldsymbol{\theta})
+\lambda,\mathrm{sign}(\boldsymbol{\theta})
$$

At zero coefficients, the absolute-value penalty is not differentiable, so a subgradient is used.

The regularisation parameter $\lambda$ controls the strength of the penalty. L1 regularisation can drive some coefficients exactly to zero, resulting in a sparse model. The intercept should not be penalised.

### L2 regularization (Ridge)

Ridge regression adds an L2-norm penalty to the MSE, penalising the squared values of the model parameters:

$$
J(\boldsymbol{\theta}) =
\mathrm{MSE}(\boldsymbol{\theta}) +
\lambda \|\boldsymbol{\theta}\|_2^2
$$

where

$$
\|\boldsymbol{\theta}\|_2^2 =
\sum_{j=0}^{p-1}\theta_j^2.
$$

The gradient is

$$
\nabla_{\boldsymbol{\theta}}J =
-\frac{2}{n}\mathbf{X}^T
(\mathbf{y}-\mathbf{X}\boldsymbol{\theta})
+2\lambda\boldsymbol{\theta}
$$

The regularisation parameter $\lambda$ controls the strength of the penalty. L2 regularisation shrinks coefficients towards zero, without setting them exactly to zero. The intercept should not be penalised.


=======
# Project_2_FYS_STK_4155
>>>>>>> dd8e4d994c9300f3de48982792184ab5b83133ec
