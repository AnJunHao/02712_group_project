from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from graphvelo.tangent_space import corr_kernel as gv_corr_kernel
from graphvelo.tangent_space import cos_corr as gv_cos_corr
from graphvelo.tangent_space import (
    density_corrected_transition_matrix as gv_density_corrected_transition_matrix,
)
from graphvelo.tangent_space import _estimate_dt as gv_estimate_dt


##Estimates the time step dt based on local density for each cell.
def _estimate_dt(
    X: NDArray[np.float64], V: NDArray[np.float64], nbrs_idx: list[list[int]]
) -> NDArray[np.float64]:
    n_cells = X.shape[0]
    dt = np.zeros(X.shape[0])
    #range through each cell
    for i, ind in enumerate(nbrs_idx):
        diff = X[neighbors] - X[i]  #difference between neighbors and cell i
        local_density = np.mean(np.linalg.norm(diff, axis=1))   #local density estimation
        velocity_norm = np.linalg.norm(V[i])    #norm of velocity vector
        dt[i] = np.median(local_density / velocity_norm)  #time step estimation
    return dt[:, None]

##Compute cosine correlation between velocity vector `vi` and directions from xi to its neighbor cells Xj
def cos_corr(
    xi: NDArray[np.float64], Xj: NDArray[np.float64], vi: NDArray[np.float64]
) -> NDArray[np.float64]:
    D = Xj - xi              #direction vectors from xi to its neighbors Xj
    dist = np.linalg.norm(D, axis=1)
    dist = np.where(dist == 0, 1, dist)  # avoid division by zero
    D = D / dist[:, None]

    # Normalize velocity vector
    v_norm = np.linalg.norm(vi)
    v_norm = v_norm if v_norm != 0 else 1

    return (D @ vi) / v_norm
    

## Compute the correlation-based transition kernel for each cell.
def corr_kernel(
    X: NDArray[np.float64],
    V: NDArray[np.float64],
    nbrs: list[list[int]],
    sigma: float | None = None,
    corr_func: Callable[[NDArray, NDArray, NDArray], NDArray] = cos_corr,
    softmax_adjusted: bool = False,
) -> NDArray[np.float64]:
    n_cells = X.shape[0]
    # Softmax
    if softmax_adjusted:
        if sigma is None:
            sigma = _estimate_sigma(X, V, nbrs, corr_func)
        logging.info(f"Using sigma={sigma:.4f}")

    P = np.zeros((n_cells, n_cells), dtype=float)   # Initialize kernal matrix

     # Compute kernel values
    for i in range(n_cells):
        idx = nbrs[i]
        c = corr_func(X[i], X[idx], V[i])    

        if softmax_adjusted:
            c = np.exp(c * sigma)
            c /= c.sum()

        P[i, idx] = c

    return P
    
    

# Row-wise mean-centering of a transition matrix T (CSR sparse matrix).
def density_corrected_transition_matrix(
    T: NDArray[np.float64] | csr_matrix,
) -> csr_matrix:
    T = sp.csr_matrix(T, copy=True)
    n_rows = T.shape[0]
     
    # Mean-center each row
    for i in range(n_rows):
        row = T[i]
        idx = row.indices
        values = row.data

        centered = values - values.mean()
        T[i, idx] = centered

    return T
