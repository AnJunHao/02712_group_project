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


def _estimate_dt(
    X: NDArray[np.float64], V: NDArray[np.float64], nbrs_idx: list[list[int]]
) -> NDArray[np.float64]:
    return gv_estimate_dt(X, V, nbrs_idx)


def cos_corr(
    xi: NDArray[np.float64], Xj: NDArray[np.float64], vi: NDArray[np.float64]
) -> NDArray[np.float64]:
    return gv_cos_corr(xi, Xj, vi)


def corr_kernel(
    X: NDArray[np.float64],
    V: NDArray[np.float64],
    nbrs: list[list[int]],
    sigma: float | None = None,
    corr_func: Callable[[NDArray, NDArray, NDArray], NDArray] = cos_corr,
    softmax_adjusted: bool = False,
) -> NDArray[np.float64]:
    return gv_corr_kernel(X, V, nbrs, sigma, corr_func, softmax_adjusted)


def density_corrected_transition_matrix(
    T: NDArray[np.float64] | csr_matrix,
) -> csr_matrix:
    return gv_density_corrected_transition_matrix(T)
