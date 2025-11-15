# 02-712 Group Project

This repository contains code and resources for the 02-712 Group Project on GraphVelo.

## Major Tasks

### 1. `project/graphvelo`

**Ziyan**

- `GraphVelo` (class): Main implementation class

- `adj_to_knn()`: Convert adjacency matrix to k-nearest neighbors

**Wen**

- `mack_score()`: Calculate MACK (Mean Average Cosine Kernel) score for evaluation

### 2. `project/preprocessing`

**Claude**

- `filter_and_normalize()`: Filter and normalize scRNA-seq data
- `neighbors()`: Compute neighborhood graph of observations
- `moments()`: Compute first and second order moments for velocity estimation
- `pca()`: Principal component analysis

### 3. `project/tools`

**Luci**

- `recover_dynamics()`: Recover full kinetic dynamics of gene expression
- `latent_time()`: Compute latent time for each cell

**Xingyu**

- `velocity()`: Estimate RNA velocity using different modes (deterministic, stochastic, dynamical)

---

## Project Structure

```
02712_group_project/
├── project/                       # Main package directory
│   ├── __init__.py
│   ├── graphvelo/                # GraphVelo implementation
│   │   ├── __init__.py
│   │   └── ...
│   ├── preprocessing/            # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── ...
│   ├── tools/                    # Analysis tools
│   │   ├── __init__.py
│   │   └── ...
│   └── plotting/                 # Visualization functions
│       ├── __init__.py
│       └── ...
├── graphvelo_scv.ipynb           # Main notebook for GraphVelo analysis
├── test_graphvelo_scv.ipynb      # Testing notebook
├── tutorial_for_scvelo.ipynb     # scVelo tutorial
├── pyproject.toml                # Project dependencies and configuration
├── uv.lock                       # Locked dependency versions
└── README.md                     # This file
```

---

## Installation

This project uses `uv` to manage the python environment and dependencies. See [uv documentation](https://docs.astral.sh/uv/) for more details.

**1. Install uv:**

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and install:**

```bash
git clone https://github.com/AnJunHao/02712_group_project.git
cd 02712_group_project
uv sync
```

Done! `uv sync` will create a virtual environment `.venv` with correct python version and all dependencies installed. `uv` uses the `pyproject.toml` file to resolve dependencies.

---

**Using the environment:**

- Option 1: Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

- Option 2: Run Jupyter Lab:

```bash
uv run python -m ipykernel install --user --name graphvelo --display-name "Python3.12 (GraphVelo)"
uv run jupyter lab
```

In Jupyter Lab, select the kernel "Python3.12 (GraphVelo)" for your notebooks.

- Option 3: Use your favorite IDE and select the python interpreter from the `.venv` folder.
Hachimi
Manbo