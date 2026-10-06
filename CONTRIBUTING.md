# How we work in this repo

## The one rule
**Code that does the work goes in `Code/src/`. Notebooks only call it.**
If you write a function in a notebook and someone else needs it, move it to `src/`.

## File ownership

| File | Owner |
|---|---|
| `src/network.py` | Person 1 |
| `src/optimizers.py` | Person 2 |
| `src/functions.py`, `src/plotting.py`, `src/data.py` (Runge) | Person 4 |
| `src/metrics.py`, `src/data.py` (MNIST) | Person 3 |
| `notebooks/part_b_regression.ipynb` | Person 2 |
| `notebooks/part_c_activations.ipynb` | Person 1 |
| `notebooks/part_d_regularization.ipynb` | Person 4 |
| `notebooks/part_e_mnist.ipynb` | Person 3 |

Only the owner edits a file. Others suggest changes through an issue or a small pull request.

## Agreed interface (don't change without telling the group)

- Network parameters: list of `(W, b)`, `W` shape `(n_in, n_out)`, forward pass `a @ W + b`.
- `backprop` returns gradients in the same structure.
- Optimizers: `step(params, grads)`.
- Adding an optional argument is fine. Renaming or reordering arguments is not.

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
