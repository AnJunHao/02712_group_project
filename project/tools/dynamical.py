# type: ignore
"""
Utilities for estimating RNA velocities under the dynamical model.

The functions in this module focus on the minimal subset that we need for the
group project: extracting kinetic parameters that were learned via
``recover_dynamics`` and computing analytical derivatives of the underlying ODE
solutions.  The implementation mirrors the mathematical model from scVelo but
is intentionally lightweight and only depends on ``AnnData`` and ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class DynamicalParameters:
    """
    Container with per-gene kinetic parameters learned by ``recover_dynamics``.

    Attributes
    ----------
    alpha, beta, gamma
        Production, splicing, and degradation rates.
    scaling
        Gene-specific scaling that was applied to match unspliced counts.
    t_switch
        Switching time between induction and repression.
    u0, s0
        Initial conditions for unspliced and spliced abundance.
    """

    alpha: FloatArray
    beta: FloatArray
    gamma: FloatArray
    scaling: FloatArray
    t_switch: FloatArray
    u0: FloatArray
    s0: FloatArray


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def extract_dynamical_parameters(
    adata: AnnData, mask: np.ndarray
) -> DynamicalParameters:
    """
    Read the parameters written by ``recover_dynamics`` and store them in a
    structured object.

    Parameters
    ----------
    adata
        Annotated data matrix containing the ``fit_*`` columns.
    mask
        Boolean mask selecting the genes for which velocities are computed.

    Returns
    -------
    DynamicalParameters
        Ready-to-use kinetic parameters.
    """

    var = adata.var.loc[:, :]
    cols = [
        "fit_alpha",
        "fit_beta",
        "fit_gamma",
        "fit_scaling",
        "fit_t_",
        "fit_u0",
        "fit_s0",
    ]
    missing = [key for key in cols if key not in var.columns]
    if missing:
        raise KeyError(
            "recover_dynamics needs to run before velocity estimation. "
            f"Missing columns: {', '.join(missing)}"
        )

    def _col(name: str) -> FloatArray:
        return np.asarray(var[name].values[mask], dtype=np.float64)

    return DynamicalParameters(
        alpha=_col("fit_alpha"),
        beta=_col("fit_beta"),
        gamma=_col("fit_gamma"),
        scaling=_col("fit_scaling"),
        t_switch=_col("fit_t_"),
        u0=_col("fit_u0"),
        s0=_col("fit_s0"),
    )


def select_velocity_genes(
    adata: AnnData,
    *,
    min_likelihood: float = 0.1,
    min_r2: float | None = None,
) -> np.ndarray:
    """
    Choose genes that should contribute to the dynamical velocity estimate.

    The routine follows scVelo's defaults: ignore genes without successful
    fits, enforce minimum likelihood and optionally minimum :math:`R^2`.

    Parameters
    ----------
    adata
        Annotated data matrix containing ``fit_alpha`` / ``fit_likelihood``.
    min_likelihood
        Likelihood threshold copied from the original implementation.
    min_r2
        Optional threshold on ``fit_r2`` (if the column exists).

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_vars,)``.
    """

    if "fit_alpha" not in adata.var.columns:
        raise ValueError(
            "recover_dynamics has to be executed before running dynamical velocity."
        )
    mask = np.isfinite(adata.var["fit_alpha"].to_numpy())

    if "fit_likelihood" in adata.var.columns and min_likelihood is not None:
        likelihood = np.asarray(adata.var["fit_likelihood"].values)
        mask &= likelihood >= min_likelihood

    if (
        min_r2 is not None
        and "fit_r2" in adata.var.columns
        and np.isfinite(adata.var["fit_r2"]).any()
    ):
        mask &= np.asarray(adata.var["fit_r2"].values) >= min_r2

    if "fit_scaling" in adata.var.columns:
        scaling = np.asarray(adata.var["fit_scaling"].values)
        if np.isfinite(scaling).any():
            lb, ub = np.nanpercentile(scaling, [10, 90])
            mask &= (scaling >= np.min([lb, 0.03])) & (scaling <= np.max([ub, 3.0]))

    return mask


# ---------------------------------------------------------------------------
# Dynamical solution helpers
# ---------------------------------------------------------------------------


def _repeat_param(values: FloatArray, n_cells: int) -> FloatArray:
    """Expand a one-dimensional per-gene array to match ``(n_cells, n_genes)``."""

    arr = np.asarray(values, dtype=np.float64).reshape(1, -1)
    return np.repeat(arr, n_cells, axis=0)


def _safe_ratio(num: FloatArray, denom: FloatArray) -> FloatArray:
    """Compute ``num / denom`` while avoiding division by zero."""

    result = np.zeros_like(num, dtype=np.float64)
    mask = np.abs(denom) > 1e-8
    result[mask] = num[mask] / denom[mask]
    return result


def _coupled_term(
    beta: FloatArray,
    gamma: FloatArray,
    value: FloatArray,
    t: FloatArray,
    exp_beta: FloatArray | None = None,
    exp_gamma: FloatArray | None = None,
) -> FloatArray:
    """
    Helper for the second summand in the spliced solution.

    The closed form becomes unstable when ``beta`` is close to ``gamma``.
    In that case we use the analytic limit :math:`\\beta \\cdot value \\cdot t`
    multiplied with :math:`e^{-\\beta t}`.
    """

    if exp_beta is None:
        exp_beta = np.exp(-beta * t)
    if exp_gamma is None:
        exp_gamma = np.exp(-gamma * t)

    term = np.empty_like(t, dtype=np.float64)
    diff = gamma - beta
    same_rate = np.abs(diff) < 1e-8

    if np.any(~same_rate):
        mask = ~same_rate
        term[mask] = (
            beta[mask]
            * value[mask]
            / diff[mask]
            * (exp_beta[mask] - exp_gamma[mask])
        )

    if np.any(same_rate):
        mask = same_rate
        term[mask] = beta[mask] * value[mask] * t[mask] * exp_beta[mask]

    return term


def _solve_induction(
    alpha: FloatArray,
    beta: FloatArray,
    gamma: FloatArray,
    t: FloatArray,
    u0: FloatArray,
    s0: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Closed-form solution for the induction phase."""

    t_clipped = np.clip(t, 0.0, None)
    exp_beta = np.exp(-beta * t_clipped)
    exp_gamma = np.exp(-gamma * t_clipped)

    u_inf = _safe_ratio(alpha, beta)
    s_inf = _safe_ratio(alpha, gamma)

    u = u_inf + (u0 - u_inf) * exp_beta
    s = s_inf + (s0 - s_inf) * exp_gamma
    s += _coupled_term(beta, gamma, u0 - u_inf, t_clipped, exp_beta, exp_gamma)
    return u, s


def _solve_repression(
    beta: FloatArray,
    gamma: FloatArray,
    dt: FloatArray,
    u_switch: FloatArray,
    s_switch: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Closed-form solution for the repression (after ``t_switch``)."""

    dt_clipped = np.clip(dt, 0.0, None)
    exp_beta = np.exp(-beta * dt_clipped)
    exp_gamma = np.exp(-gamma * dt_clipped)

    u = u_switch * exp_beta
    s = s_switch * exp_gamma
    s += _coupled_term(beta, gamma, u_switch, dt_clipped, exp_beta, exp_gamma)
    return u, s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_dynamical_velocity(
    adata: AnnData,
    params: DynamicalParameters,
    latent_time: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Evaluate the dynamical system for each cell and gene and return derivatives.

    Parameters
    ----------
    adata
        Annotated data matrix (used for the number of cells).
    params
        Kinetic parameters for the selected genes.
    latent_time
        Matrix ``(n_obs, n_genes)`` with cell-specific latent timepoints.

    Returns
    -------
    tuple of np.ndarray
        ``(velocity_spliced, velocity_unspliced)``, each dense float64 arrays.
        Entries corresponding to missing latent timepoints are filled with NaN.
    """

    if latent_time.ndim != 2:
        raise ValueError("latent_time must be a 2-D matrix of shape (n_obs, n_genes)")

    n_cells, _ = latent_time.shape

    alpha = _repeat_param(params.alpha, n_cells)
    beta = _repeat_param(params.beta, n_cells)
    gamma = _repeat_param(params.gamma, n_cells)
    scaling = _repeat_param(params.scaling, n_cells)
    u0 = _repeat_param(params.u0, n_cells)
    s0 = _repeat_param(params.s0, n_cells)
    t_switch = _repeat_param(params.t_switch, n_cells)

    latent = np.asarray(latent_time, dtype=np.float64)
    valid = np.isfinite(latent)
    latent = np.where(valid, np.clip(latent, 0.0, None), 0.0)

    u_ind, s_ind = _solve_induction(alpha, beta, gamma, latent, u0, s0)

    switch_u, switch_s = _solve_induction(
        alpha[:1, :], beta[:1, :], gamma[:1, :], t_switch[:1, :], u0[:1, :], s0[:1, :]
    )
    switch_u = np.repeat(switch_u, n_cells, axis=0)
    switch_s = np.repeat(switch_s, n_cells, axis=0)

    dt = np.maximum(latent - t_switch, 0.0)
    u_rep, s_rep = _solve_repression(beta, gamma, dt, switch_u, switch_s)

    induction = latent <= t_switch

    u = np.where(induction, u_ind, u_rep)
    s = np.where(induction, s_ind, s_rep)

    du_dt = np.where(induction, alpha - beta * u, -beta * u)
    ds_dt = beta * u - gamma * s

    du_dt = du_dt * scaling

    du_dt[~valid] = np.nan
    ds_dt[~valid] = np.nan

    return ds_dt, du_dt
