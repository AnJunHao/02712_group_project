from project import preprocessing as pp
from project import tools as tl

# Conditional imports for modules that may have missing dependencies
try:
    from project import graphvelo as gv
except (ImportError, ModuleNotFoundError):
    gv = None  # graphvelo module not available (missing dependencies)

try:
    from project import plotting as pl
except (ImportError, ModuleNotFoundError):
    pl = None  # plotting module not available (missing dependencies)

__all__ = ["pp", "tl", "gv", "pl"]
