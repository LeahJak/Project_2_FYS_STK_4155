"""
Shared plotting style and helpers, so all figures in the report look the same.
The style is applied automatically when this file is imported.
Owner: Person 4

Usage
-----
    from src.plotting import line_plot, heatmap, save_fig

Every helper returns (fig, ax). Give name="..." to save the figure as
figures/<name>.pdf; leave it out to only draw it. Name figures by part,
e.g. "b_fit", "b_lr_sweep", "d_heatmap_l2".

1. Fitted curves against the data and the true function (x on the original scale):

    x_plot = np.linspace(-1, 1, 300).reshape(-1, 1)
    line_plot(x_plot,
              {"true function": runge(x_plot),
               "OLS, degree 8": y_ols_plot,
               "FFNN 1x50": net.predict((x_plot - mean) / std)},
              "$x$", "$y$", data=(X_train, y_train), name="b_fit")

2. Learning curve from NeuralNetwork.train, with the noise floor sigma^2:

    history = net.train(X_train_s, y_train, epochs=1000, batch_size=16, rng=rng)
    line_plot(np.arange(1, len(history) + 1), {r"Adam, $\\gamma$ = 0.01": history},
              "Epoch", "Training MSE", logy=True, hlines={"noise floor": 0.01},
              name="b_learning_curve")

3. Test MSE against the learning rate, with OLS as reference:

    line_plot(gammas, {"GD": mse_gd, "RMSprop": mse_rms, "Adam": mse_adam},
              r"Learning rate $\\gamma$", "Test MSE", logx=True, logy=True,
              hlines={"OLS": mse_ols}, name="b_lr_sweep")

4. Heat map of two hyperparameters. Z[i, j] belongs to y_values[i] and x_values[j]:

    Z = np.zeros((len(lambdas), len(gammas)))
    for i, lmbda in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            Z[i, j] = ...           # test MSE for this (lambda, gamma)
    Z[Z > 1] = np.nan               # diverged runs shown as blank cells
    heatmap(Z, gammas, lambdas, r"$\\gamma$", r"$\\lambda$", name="d_heatmap_l2")

5. Adjusting a figure before saving: leave out name, change it, then save:

    fig, ax = line_plot(...)
    ax.set_ylim(1e-3, 1)
    save_fig("b_lr_sweep")

6. Any other plot gets the same style automatically:

    plt.plot(x, y)
    save_fig("c_something")

LLM-assisted
------------
Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
Role: Wrote the style settings, save_fig, line_plot, heatmap and the usage
examples above. The design (style applied on import, saving by figure name,
Seaborn for the heat map, colours from project 1) was decided by the group.
Modifications/verification: none yet. 
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Figures are saved in <repo>/figures, found from this file's location (<repo>/Code/src/plotting.py),
# so it works no matter which folder a notebook or script is run from.
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

# Colours from project 1 (colour-blind safe)
BLUE, RED, YELLOW, GREEN, PINK = "#004488", "#BB5566", "#DDAA33", "#29C44F", "#DD338B"
COLORS = [BLUE, RED, YELLOW, GREEN, PINK]

# Shared style, applied to every figure made after this file is imported
plt.rcParams.update({
    "figure.figsize": (6, 4),                        # size of one figure in the report
    "font.size": 11,                                 # readable text after scaling into the PDF
    "legend.frameon": False,                         # no box around legends
    "axes.grid": True,                               # faint grid in every plot
    "grid.alpha": 0.3,
    "savefig.dpi": 300,                              # resolution of saved PNGs
    "savefig.bbox": "tight",                         # trim empty margins when saving
    "axes.prop_cycle": plt.cycler(color=COLORS),     # line colours in this order
})


def save_fig(name):
    """
    Save the current figure as <repo>/figures/<name>.pdf.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.pdf")


def line_plot(x, lines, xlabel, ylabel, name=None, logx=False, logy=False,
              data=None, hlines=None):
    """
    One or more lines against the same x values.

    Parameters
    ----------
    x      : x values shared by all lines.
    lines  : dict {label: y values}, e.g. {"Adam": mse_adam, "RMSprop": mse_rms}.
    xlabel, ylabel : axis labels.
    name   : if given, the figure is saved as figures/<name>.pdf.
    logx, logy : logarithmic axes.
    data   : optional (x_points, y_points) shown as grey dots, e.g. the training data.
    hlines : optional dict {label: value} for horizontal reference lines, e.g. {"OLS": mse_ols}.

    Returns
    -------
    fig, ax

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function from our list of plot types (fits, learning curves, sweeps).
    """
    fig, ax = plt.subplots()
    if data is not None:
        ax.scatter(np.ravel(data[0]), np.ravel(data[1]), s=12, color="gray", alpha=0.6, label="data")
    for label, y in lines.items():
        ax.plot(np.ravel(x), np.ravel(y), label=label)
    if hlines is not None:
        for label, value in hlines.items():
            ax.axhline(value, color="black", ls="--", lw=1, label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if name is not None:
        save_fig(name)
    return fig, ax


def heatmap(Z, x_values, y_values, xlabel, ylabel, label="Test MSE", name=None):
    """
    Heat map of a score over a grid of two hyperparameters, value written in each cell.

    Parameters
    ----------
    Z        : 2D array with shape (len(y_values), len(x_values)). NaN cells
               (e.g. diverged runs) are left blank.
    x_values, y_values : hyperparameter values along the axes.
    xlabel, ylabel : axis labels.
    label    : colour bar label.
    name     : if given, the figure is saved as figures/<name>.pdf.

    Returns
    -------
    fig, ax

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function using seaborn.heatmap, as suggested in the project text.
    """
    fig, ax = plt.subplots()
    sns.heatmap(Z, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                xticklabels=[f"{v:g}" for v in x_values],
                yticklabels=[f"{v:g}" for v in y_values],
                cbar_kws={"label": label})
    ax.invert_yaxis()   # smallest y value at the bottom
    ax.grid(False)      # the global grid would draw lines across the cells
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if name is not None:
        save_fig(name)
    return fig, ax