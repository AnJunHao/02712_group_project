from __future__ import annotations

from typing import Any, Literal

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray

from project.tools.dynamical import (
    compute_dynamical_velocity,
    extract_dynamical_parameters,
    select_velocity_genes,
)
from project.tools.latent import infer_latent_time
from scvelo.tools import recover_dynamics as scv_recover_dynamics


def recover_dynamics(
    data: AnnData,
    var_names: str | list[str] = "velocity_genes",
    n_top_genes: int | None = None,
    max_iter: int = 10,
    assignment_mode: str = "projection",
    t_max: float | bool | None = None,
    fit_time: bool | float | None = True,
    fit_scaling: bool | float | None = True,
    fit_steady_states: bool | None = True,
    fit_connected_states: bool | None = None,
    fit_basal_transcription: bool | None = None,
    use_raw: bool = False,
    load_pars: bool | None = None,
    return_model: bool | None = None,
    plot_results: bool = False,
    steady_state_prior: list[bool] | None = None,
    add_key: str = "fit",
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> AnnData | None:
    return scv_recover_dynamics(
        data=data,
        var_names=var_names,  # type: ignore
        n_top_genes=n_top_genes,
        max_iter=max_iter,
        assignment_mode=assignment_mode,
        t_max=t_max,
        fit_time=fit_time,  # type: ignore
        fit_scaling=fit_scaling,  # type: ignore
        fit_steady_states=fit_steady_states,  # type: ignore
        fit_connected_states=fit_connected_states,
        fit_basal_transcription=fit_basal_transcription,
        use_raw=use_raw,
        load_pars=load_pars,
        return_model=return_model,
        plot_results=plot_results,
        steady_state_prior=steady_state_prior,
        add_key=add_key,
        copy=copy,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
        **kwargs,
    )


def velocity(
    data: AnnData,
    vkey: str = "velocity",
    mode: Literal["dynamical", "stochastic"] = "dynamical",
    fit_offset: bool = False,
    fit_offset2: bool = False,
    filter_genes: bool = False,
    groups: str | list | None = None,
    groupby: str | list | NDArray | None = None,
    groups_for_fit: str | list | NDArray | None = None,
    constrain_ratio: float | tuple[float, ...] | None = None,
    use_raw: bool = False,
    use_latent_time: bool | None = None,
    perc: float | None = None,
    min_r2: float = 1e-2,
    min_likelihood: float = 1e-3,
    r2_adjusted: bool | None = None,
    use_highly_variable: bool = True,
    diff_kinetics: Any = None,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData | None:
    """
    Compute RNA velocities using the dynamical model.

    This reimplementation focuses solely on the dynamical mode and follows the
    derivation used in scVelo: kinetic parameters must be present (produced by
    ``recover_dynamics``) and we analytically evaluate the ODE solution for each
    cell/gene pair to obtain `ds/dt` and `du/dt`.
    """

    if mode != "dynamical":
        raise NotImplementedError(
            "Only the dynamical model is supported in this implementation."
        )
    if fit_offset or fit_offset2 or groups or groupby or groups_for_fit:
        raise NotImplementedError(
            "Group-specific or offset fits are not implemented in this simplified "
            "dynamical velocity."
        )

    adata = data.copy() if copy else data

    if "fit_t" not in adata.layers:
        raise ValueError("recover_dynamics must be executed before velocity().")

    gene_mask = select_velocity_genes(
        adata, min_likelihood=min_likelihood, min_r2=min_r2
    )
    if not np.any(gene_mask):
        raise ValueError("No genes passed the velocity filtering thresholds.")

    params = extract_dynamical_parameters(adata, gene_mask)
    latent = np.asarray(adata.layers["fit_t"][:, gene_mask], dtype=np.float64)
    v_spliced, v_unspliced = compute_dynamical_velocity(adata, params, latent)

    # Fill result layers with NaNs and write subset
    velocity_layer = np.full(adata.shape, np.nan, dtype=np.float32)
    velocity_layer[:, gene_mask] = v_spliced.astype(np.float32)
    adata.layers[vkey] = velocity_layer

    velocity_u = np.full(adata.shape, np.nan, dtype=np.float32)
    velocity_u[:, gene_mask] = v_unspliced.astype(np.float32)
    adata.layers[f"{vkey}_u"] = velocity_u

    adata.var[f"{vkey}_genes"] = gene_mask
    adata.uns[f"{vkey}_params"] = {
        "mode": mode,
        "fit_offset": fit_offset,
        "fit_offset2": fit_offset2,
        "min_r2": min_r2,
        "min_likelihood": min_likelihood,
    }

    if filter_genes and np.sum(gene_mask) > 0:
        adata._inplace_subset_var(gene_mask)

    return adata if copy else None


def latent_time(
    data: AnnData,
    vkey: str = "velocity",
    min_likelihood: float = 0.1,
    min_confidence: float = 0.75,
    min_corr_diffusion: float | None = None,
    weight_diffusion: float | None = None,
    root_key: str | None = None,
    end_key: str | None = None,
    t_max: float | None = None,
    copy: bool = False,
) -> AnnData | None:
    """
    Compute a gene-shared latent time using recovered dynamics.

    The estimator aggregates gene-specific latent times (``fit_t``) using
    likelihood weighting and enforces smoothness by propagating times across
    the neighborhood connectivities graph.
    """

    if min_corr_diffusion is not None:
        raise NotImplementedError(
            "min_corr_diffusion is not supported in the lightweight latent time."
        )

    adata = data.copy() if copy else data

    gene_mask = select_velocity_genes(
        adata, min_likelihood=min_likelihood, min_r2=None
    )
    latent = infer_latent_time(
        adata,
        gene_mask,
        min_confidence=min_confidence,
        min_likelihood=min_likelihood,
        weight_diffusion=weight_diffusion,
        root_key=root_key,
        end_key=end_key,
        t_max=t_max,
    )

    adata.var[f"{vkey}_genes"] = gene_mask
    adata.obs["latent_time"] = latent.astype(np.float32)

    return adata if copy else None
