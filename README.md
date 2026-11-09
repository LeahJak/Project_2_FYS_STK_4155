
# FYS-STK3155/4155 Project 2: Feed-forward neural networks

Group members: Nora Arnesen, Viktoria Krogen, Natalie Kristiansen, Leah Jakobsen 

This repository contains the implementation and experiments for Project 2 in FYS-STK-4155. The project explores neural networks for regression and classification, with a focus on understanding their architecture, training algorithms and regularization methods.

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
│   ├── notebooks/      One notebook per part
│   ├── tests/          pytest tests
│   ├── pyproject.toml
│   └── requirements.txt
├── figures/            All figures produced in the experiments 
└── Report/             PDF of the final report
```

## Installation

The project uses Python and the dependencies listed in `requirements.txt`. To install them, first create and activate a virtual environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the required packages in the virtual environment. Make sure the environment is activated whenever you run the project.


## Reproducing the results

Open each notebook in `Code/notebooks/` and run **Kernel → Restart & Run All**.
Figures are saved to `figures/`.

## Running tests

The project uses `pytest` for testing. Make sure your virtual environment is activated and the required dependencies are installed.

To run all tests, execute the following command from the project root directory:

```bash
pytest
```

To run the tests with more detailed output, use:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_filename.py
```

## Use of LLMs

LLMs were used to establish the repository structure and create the initial drafts of `README.md` (Claude AI, October 2026).

For information about the use of LLMs during the project work and report writing, see [Appendix A of the report](Report/FYS-STK4155__project_2.pdf).

## File overview

| File | Content |
|---|---|
| `src/network.py` | `NeuralNetwork`: `create_layers`, `predict`, `penalty`, `cost`, `feed_forward_saver`, `compute_gradient` w/L1/L2, `update_weights`, `train` | 
| `src/optimizers.py` | `GD`, `SGD`,`RMSprop`,`Adam`| 
| `src/functions.py` | `mse`, `mse_der`, `L1`, `L1_der`, `L2`, `L2_der`, `cross_entropy`, `softmax`, `softmax_cross_entropy_der`, `sigmoid`, `sigmoid_der`, `ReLU`, `ReLU_der`, `leaky_ReLU`, `leaky_ReLU_der`, `identity`, `identity_der` | 
| `src/plotting.py` | `save_fig`, `line_plot`, `heatmap` |
| `src/data.py` | `runge`, `design_matrix`, `make_runge`, `make_runge_split`, `make_batches`, `scale_data`, `center_data`, `load_mnist`, `one_hot`, `stratified_kfold_indices`, `make_runge_split_3` |
| `src/metrics.py` | `accuracy`, `confusion_matrix_multiclass`, `per_class_from_confusion`, `standard_error`, `r2` |
| `notebooks/part_b_regression.ipynb` | NN vs OLS, learning rates |
| `notebooks/part_c_activations.ipynb` |  Activations, depth and width |
| `notebooks/part_d_regularization.ipynb` |  L1/L2 vs Ridge/Lasso |
| `notebooks/part_e_mnist.ipynb` | MNIST classification |

