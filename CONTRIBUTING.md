# How we work in this repo

## The one rule
**Code that does the work goes in `Code/src/`. Notebooks only call it.**
If you write a function in a notebook and someone else needs it, move it to `src/`.


## File overview

| File | Content | Aim |
|---|---|---|
| `src/network.py` | `NeuralNetwork`: `create_layers`, `predict`, `penalty`, `cost`, ``jax_predict`, `jax_cost`, `feed_forward_saver`, `compute_gradient` w/L1/L2, `update_weights`, `train` | :) |
| `src/optimizers.py` | `GD`, `SGD`,`RMSprop`,`Adam`|  :)  |
| `src/functions.py` | `mse`, `mse_der`, `L1`, `L1_der`, `L2`, `L2_der`, `cross_entropy`, `softmax`, `softmax_cross_entropy_der`, `sigmoid`, `sigmoid_der`, `ReLU`, `ReLU_der`, `leaky_ReLU`, `leaky_ReLU_der`, `identity`, `identity_der` | `binary_cross_entropy`, `binary_cross_entropy_der` |
| `src/plotting.py` | - | `heatmap`, `savefig` |
| `src/data.py` | `runge`, `design_matrix`, `make_runge`, `make_runge_split`, `make_batches` | Scaling, MNIST loading |
| `src/metrics.py` | - | `r2`, `accuracy`, `confusion_matrix` |
| `notebooks/part_b_regression.ipynb` | - | NN vs OLS, learning rates |
| `notebooks/part_c_activations.ipynb` | - | Activations, depth and width |
| `notebooks/part_d_regularization.ipynb` | - | L1/L2 vs Ridge/Lasso |
| `notebooks/part_e_mnist.ipynb` | - | MNIST classification |


## Agreed interface (don't change without telling the group)

- Network parameters: list of `(W, b)`, `W` shape `(n_in, n_out)`, forward pass `a @ W + b`.
- `backprop` returns gradients in the same structure.
- Optimizers: `update(layers, layer_grads)`.
- Adding an optional argument is fine. Renaming or reordering arguments is not.
- Cost functions and metrics take arguments in the order `(predict, target)`.

## Daily git workflow

```bash
git checkout main
git pull                              # get everyone's latest work
git checkout -b p2-adam               # one branch per task, prefix with your number
# ... work, commit often ...
git pull origin main                  # bring in others' changes, fix conflicts early
git push -u origin p2-adam
```

Then open a pull request on GitHub.

- Never commit directly to `main`.
- Keep branches short: merge every day or two.
- Pull `main` into your branch often.

## Notebooks

- Start every notebook with `%load_ext autoreload` and `%autoreload 2`.
- Before pushing: **Kernel → Restart & Run All** must run without errors.
- Save report figures to `figures/`.
- Outputs are stripped automatically by nbstripout.
- On a notebook merge conflict, run `git mergetool --tool=nbdime` locally. Don't use GitHub's web editor.

## Scratch work

Throwaway notebooks go in a folder called `scratch/`. It is in `.gitignore`.
