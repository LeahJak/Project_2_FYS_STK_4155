<<<<<<< HEAD
# FYS-STK3155/4155 Project 2: Feed-forward neural networks

Group members: _add names_

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
│   │   └── plotting.py     shared plot style and helpers
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
=======
# Project_2_FYS_STK_4155
>>>>>>> dd8e4d994c9300f3de48982792184ab5b83133ec
