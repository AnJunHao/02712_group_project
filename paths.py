from pathlib import Path

# Root of your project
PROJECT_DIR = Path(__file__).resolve().parent

# Directory where your data is stored
DATA_DIR = PROJECT_DIR / "data"

# Directory where your figures should be saved
FIG_DIR = PROJECT_DIR / "figures"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)