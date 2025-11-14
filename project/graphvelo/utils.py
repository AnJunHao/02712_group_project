import numpy as np
import pandas as pd
from numpy.typing import NDArray
from anndata import AnnData

from graphvelo.utils import adj_to_knn as gv_adj_to_knn
from graphvelo.utils import mack_score as gv_mack_score


def adj_to_knn(
    adj: NDArray[np.float64], n_neighbors: int = 30
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    return gv_adj_to_knn(adj, n_neighbors)


def mack_score(
    adata: AnnData,
    n_neighbors: int | None = None,
    basis: str | None = None,
    tkey: str | None = None,
    genes: list | None = None,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    X_data: NDArray[np.float64] | None = None,
    V_data: NDArray[np.float64] | None = None,
    n_jobs: int = -1,
    add_prefix: str | None = None,
    return_score: bool = False,
) -> pd.DataFrame | None:
    return gv_mack_score(
        adata,
        n_neighbors,
        basis,
        tkey,
        genes,
        ekey,
        vkey,
        X_data,
        V_data,
        n_jobs,
        add_prefix,
        return_score,
    )
