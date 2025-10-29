# 02-712 Group Project

This repository contains code and resources for the 02-712 Group Project on GraphVelo.

## Installation

**Install uv:**
```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Clone and install:**
```bash
git clone https://github.com/AnJunHao/02712_group_project.git
cd 02712_group_project
uv sync
```

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
