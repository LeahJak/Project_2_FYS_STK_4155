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
│   ├── notebooks/      One notebook per part
│   ├── tests/          pytest tests
│   ├── pyproject.toml
│   └── requirements.txt
├── figures/            All figures produced in the experiments 
└── Report/             PDF of the final report
```


## Reproducing the results

Open each notebook in `Code/notebooks/` and run **Kernel → Restart & Run All**.
Figures are saved to `figures/`.

## Install the required packages 

```bash 
pip install -r requirements.txt
```

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

LLM's were used to create the repository layout, as well as the fundaments for the README.md and CONTRIBUTING.md (Claude AI, October 2026). 

## File overview

| File | Content | Aim |
|---|---|---|
| `src/network.py` | `NeuralNetwork`: `create_layers`, `predict`, `penalty`, `cost`, `feed_forward_saver`, `compute_gradient` w/L1/L2, `update_weights`, `train` | :) |
| `src/optimizers.py` | `GD`, `SGD`,`RMSprop`,`Adam`|  :)  |
| `src/functions.py` | `mse`, `mse_der`, `L1`, `L1_der`, `L2`, `L2_der`, `cross_entropy`, `softmax`, `softmax_cross_entropy_der`, `sigmoid`, `sigmoid_der`, `ReLU`, `ReLU_der`, `leaky_ReLU`, `leaky_ReLU_der`, `identity`, `identity_der` | `binary_cross_entropy`, `binary_cross_entropy_der` |
| `src/plotting.py` | `save_fig`, `line_plot`, `heatmap` | - |
| `src/data.py` | `runge`, `design_matrix`, `make_runge`, `make_runge_split`, `make_batches`, `scale_data`, `center_data`, `load_mnist`, `one_hot`, `make_mnist_split` | - |
| `src/metrics.py` | `accuracy`, `confusion_matrix_multiclass`, `per_class_from_confusion`, `standard_error`, `r2` | `confusion_matrix` |
| `notebooks/part_b_regression.ipynb` | - | NN vs OLS, learning rates |
| `notebooks/part_c_activations.ipynb` | - | Activations, depth and width |
| `notebooks/part_d_regularization.ipynb` | - | L1/L2 vs Ridge/Lasso |
| `notebooks/part_e_mnist.ipynb` | - | MNIST classification |



=======
# Project_2_FYS_STK_4155
>>>>>>> dd8e4d994c9300f3de48982792184ab5b83133ec
