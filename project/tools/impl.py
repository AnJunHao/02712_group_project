from __future__ import annotations

from typing import Any, Literal
import os

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
import matplotlib.pyplot as pl
from matplotlib import rcParams

from project.tools.dynamical import (
    compute_dynamical_velocity,
    extract_dynamical_parameters,
    select_velocity_genes,
)
from project.tools.latent import infer_latent_time
from project.tools._dynamics_recovery import DynamicsRecovery
from project.tools.utils import make_unique_list

from scvelo import logging as logg
from scvelo import settings
from scvelo.core import get_n_jobs, parallelize
from scvelo.preprocessing.moments import get_connectivities


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


default_pars_names = ["alpha", "beta", "gamma", "t_", "scaling", "std_u", "std_s"]
default_pars_names += ["likelihood", "u0", "s0", "pval_steady"]
default_pars_names += ["steady_u", "steady_s", "variance"]


def _read_pars(adata, pars_names=None, key="fit"):
    """Read parameters from adata.var."""
    pars = []
    for name in default_pars_names if pars_names is None else pars_names:
        pkey = f"{key}_{name}"
        par = np.zeros(adata.n_vars) * np.nan
        if pkey in adata.var.keys():
            par = adata.var[pkey].values
        pars.append(par)
    return pars


def _write_pars(adata, pars, pars_names=None, add_key="fit"):
    """Write parameters to adata.var."""
    for i, name in enumerate(default_pars_names if pars_names is None else pars_names):
        adata.var[f"{add_key}_{name}"] = pars[i]


def _fit_recovery(
    var_names,
    adata,
    use_raw,
    load_pars,
    max_iter,
    fit_time,
    fit_steady_states,
    conn,
    fit_scaling,
    fit_basal_transcription,
    steady_state_prior,
    assignment_mode,
    queue,
    **kwargs,
):
    """Fit recovery for a list of genes."""
    idx, dms = [], []
    for gene in var_names:
        dm = DynamicsRecovery(
            adata,
            gene,
            use_raw=use_raw,
            load_pars=load_pars,
            max_iter=max_iter,
            fit_time=fit_time,
            fit_steady_states=fit_steady_states,
            fit_connected_states=conn,
            fit_scaling=fit_scaling,
            fit_basal_transcription=fit_basal_transcription,
            steady_state_prior=steady_state_prior,
            **kwargs,
        )
        if dm.recoverable:
            dm.fit(assignment_mode=assignment_mode)

            ix = np.where(adata.var_names == gene)[0][0]
            idx.append(ix)
            dms.append(dm)
        else:
            logg.warn(dm.gene, "not recoverable due to insufficient samples.")

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return idx, dms


def _flatten(iterable):
    """Flatten nested lists."""
    return [i for it in iterable for i in it]


def align_dynamics(
    data, t_max=None, dm=None, idx=None, mode=None, remove_outliers=None, copy=False
):
    """Align dynamics to a common set of parameters."""
    adata = data.copy() if copy else data
    pars_names = ["alpha", "beta", "gamma", "t_", "scaling", "alignment_scaling"]
    alpha, beta, gamma, t_, scaling, mz = _read_pars(adata, pars_names=pars_names)
    T = np.zeros(adata.shape) * np.nan
    Tau = np.zeros(adata.shape) * np.nan
    Tau_ = np.zeros(adata.shape) * np.nan
    if "fit_t" in adata.layers.keys():
        T = adata.layers["fit_t"]
    if "fit_tau" in adata.layers.keys():
        Tau = adata.layers["fit_tau"]
    if "fit_tau_" in adata.layers.keys():
        Tau_ = adata.layers["fit_tau_"]
    idx = ~np.isnan(np.sum(T, axis=0)) if idx is None else idx
    if "fit_alignment_scaling" not in adata.var.keys():
        mz = np.ones(adata.n_vars)
    if mode is None:
        mode = "align_total_time"

    m = np.ones(adata.n_vars)
    mz_prev = np.array(mz)

    if dm is not None:  # newly fitted
        mz[idx] = 1

    if mode == "align_total_time" and t_max is not False:
        T_max = np.max(T[:, idx] * (T[:, idx] < t_[idx]), axis=0)
        T_max += np.max((T[:, idx] - t_[idx]) * (T[:, idx] > t_[idx]), axis=0)

        denom = 1 - np.sum((T[:, idx] == t_[idx]) | (T[:, idx] == 0), axis=0) / len(T)
        denom += denom == 0

        T_max = T_max / denom
        T_max += T_max == 0

        t_max = 20 if t_max is None else t_max
        m[idx] = t_max / T_max
        mz *= m

    else:
        m = 1 / mz
        mz = np.ones(adata.n_vars)

    if remove_outliers:
        mu, std = np.nanmean(mz), np.nanstd(mz)
        mz = np.clip(mz, mu - 3 * std, mu + 3 * std)
        m = mz / mz_prev

    if idx is None:
        alpha, beta, gamma = alpha / m, beta / m, gamma / m
        T, t_, Tau, Tau_ = T * m, t_ * m, Tau * m, Tau_ * m
    else:
        m_ = m[idx]
        alpha[idx] = alpha[idx] / m_
        beta[idx] = beta[idx] / m_
        gamma[idx] = gamma[idx] / m_
        T[:, idx], t_[idx] = T[:, idx] * m_, t_[idx] * m_
        Tau[:, idx], Tau_[:, idx] = Tau[:, idx] * m_, Tau_[:, idx] * m_

    mz[mz == 1] = np.nan
    pars_names = ["alpha", "beta", "gamma", "t_", "alignment_scaling"]
    _write_pars(adata, [alpha, beta, gamma, t_, mz], pars_names=pars_names)
    adata.layers["fit_t"] = T
    adata.layers["fit_tau"] = Tau
    adata.layers["fit_tau_"] = Tau_

    if dm is not None and dm.recoverable:
        dm.m = m[idx]
        dm.alpha = dm.alpha / dm.m[-1]
        dm.beta = dm.beta / dm.m[-1]
        dm.gamma = dm.gamma / dm.m[-1]
        dm.pars[:3] = dm.pars[:3] / dm.m[-1]

        dm.t = dm.t * dm.m[-1]
        dm.tau = dm.tau * dm.m[-1]
        dm.t_ = dm.t_ * dm.m[-1]
        dm.pars[4] = dm.pars[4] * dm.m[-1]

    return adata if copy else dm


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
    """Recovers the full splicing kinetics of specified genes.

    The model infers transcription rates, splicing rates, degradation rates,
    as well as cell-specific latent time and transcriptional states,
    estimated iteratively by expectation-maximization.
    """
    adata = data.copy() if copy else data

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(f"recovering dynamics (using {n_jobs}/{os.cpu_count()} cores)", r=True)

    if len(set(adata.var_names)) != len(adata.var_names):
        logg.warn("Duplicate var_names found. Making them unique.")
        adata.var_names_make_unique()

    if "Ms" not in adata.layers.keys() or "Mu" not in adata.layers.keys():
        use_raw = True
    if fit_connected_states is None:
        fit_connected_states = not use_raw

    adata.uns["recover_dynamics"] = {
        "fit_connected_states": fit_connected_states,
        "fit_basal_transcription": fit_basal_transcription,
        "use_raw": use_raw,
    }

    if isinstance(var_names, str) and var_names not in adata.var_names:
        if var_names in adata.var.keys():
            var_names = adata.var_names[adata.var[var_names].values]
        elif use_raw or var_names == "all":
            var_names = adata.var_names
        elif "_genes" in var_names:
            # Use scVelo's Velocity class for gene selection
            from scvelo.tools.velocity import Velocity

            velo = Velocity(adata, use_raw=use_raw)
            velo.compute_deterministic(perc=[5, 95])
            var_names = adata.var_names[velo._velocity_genes]
            adata.var["fit_r2"] = velo._r2
        else:
            raise ValueError("Variable name not found in var keys.")
    if not isinstance(var_names, str):
        var_names = list(np.ravel(var_names))

    var_names = make_unique_list(var_names, allow_array=True)
    var_names = np.array([name for name in var_names if name in adata.var_names])
    if len(var_names) == 0:
        raise ValueError("Variable name not found in var keys.")
    if n_top_genes is not None and len(var_names) > n_top_genes:
        X = adata[:, var_names].layers[("spliced" if use_raw else "Ms")]
        var_names = var_names[np.argsort(np.sum(X, 0))[::-1][:n_top_genes]]
    if return_model is None:
        return_model = len(var_names) < 5

    pars = _read_pars(adata)
    alpha, beta, gamma, t_, scaling, std_u, std_s, likelihood = pars[:8]
    u0, s0, pval, steady_u, steady_s, varx = pars[8:]
    idx, L, P = [], [], []
    T = np.zeros(adata.shape) * np.nan
    Tau = np.zeros(adata.shape) * np.nan
    Tau_ = np.zeros(adata.shape) * np.nan
    if "fit_t" in adata.layers.keys():
        T = adata.layers["fit_t"]
    if "fit_tau" in adata.layers.keys():
        Tau = adata.layers["fit_tau"]
    if "fit_tau_" in adata.layers.keys():
        Tau_ = adata.layers["fit_tau_"]

    conn = get_connectivities(adata) if fit_connected_states else None

    res = parallelize(
        _fit_recovery,
        var_names,
        n_jobs,
        unit="gene",
        as_array=False,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        adata=adata,
        use_raw=use_raw,
        load_pars=load_pars,
        max_iter=max_iter,
        fit_time=fit_time,
        fit_steady_states=fit_steady_states,
        fit_scaling=fit_scaling,
        fit_basal_transcription=fit_basal_transcription,
        steady_state_prior=steady_state_prior,
        conn=conn,
        assignment_mode=assignment_mode,
        **kwargs,
    )
    idx, dms = map(_flatten, zip(*res))

    for ix, dm in zip(idx, dms):
        T[:, ix], Tau[:, ix], Tau_[:, ix] = dm.t, dm.tau, dm.tau_
        alpha[ix], beta[ix], gamma[ix], t_[ix], scaling[ix] = dm.pars[:, -1]
        u0[ix], s0[ix], pval[ix] = dm.u0, dm.s0, dm.pval_steady
        steady_u[ix], steady_s[ix] = dm.steady_u, dm.steady_s
        beta[ix] /= scaling[ix]
        steady_u[ix] *= scaling[ix]

        std_u[ix], std_s[ix] = dm.std_u, dm.std_s
        likelihood[ix], varx[ix] = dm.likelihood, dm.varx
        L.append(dm.loss)

    _pars = [
        alpha,
        beta,
        gamma,
        t_,
        scaling,
        std_u,
        std_s,
        likelihood,
        u0,
        s0,
        pval,
        steady_u,
        steady_s,
        varx,
    ]
    _write_pars(adata, _pars)
    if "fit_t" in adata.layers.keys():
        adata.layers["fit_t"][:, idx] = (
            T[:, idx] if conn is None else conn.dot(T[:, idx])
        )
    else:
        adata.layers["fit_t"] = T if conn is None else conn.dot(T)
    adata.layers["fit_tau"] = Tau
    adata.layers["fit_tau_"] = Tau_

    if L:
        cur_len = adata.varm["loss"].shape[1] if "loss" in adata.varm.keys() else 2
        max_len = max(np.max([len(loss) for loss in L]), cur_len) if L else cur_len
        loss = np.ones((adata.n_vars, max_len)) * np.nan

        if "loss" in adata.varm.keys():
            loss[:, :cur_len] = adata.varm["loss"]

        loss[idx] = np.vstack(
            [
                np.concatenate([loss, np.ones(max_len - len(loss)) * np.nan])
                for loss in L
            ]
        )
        adata.varm["loss"] = loss

    if t_max is not False:
        dm = align_dynamics(adata, t_max=t_max, dm=dm, idx=idx)

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{add_key}_pars', "
        f"fitted parameters for splicing dynamics (adata.var)"
    )

    if plot_results:
        n_rows, n_cols = len(var_names[:4]), 6
        figsize = [2 * n_cols, 1.5 * n_rows]
        fontsize = rcParams["font.size"]
        fig, axes = pl.subplots(nrows=n_rows, ncols=6, figsize=figsize)
        pl.subplots_adjust(wspace=0.7, hspace=0.5)
        for var_id in range(4):
            if t_max is not False:
                mi = dm.m[var_id]
                P[var_id] *= np.array([1 / mi, 1 / mi, 1 / mi, mi, 1])[:, None]
            ax = axes[var_id] if n_rows > 1 else axes
            for j, pij in enumerate(P[var_id]):
                ax[j].plot(pij)
            ax[len(P[var_id])].plot(L[var_id])
            if var_id == 0:
                pars_names_list = ["alpha", "beta", "gamma", "t_", "scaling", "loss"]
                for j, name in enumerate(pars_names_list):
                    ax[j].set_title(name, fontsize=fontsize)

    if return_model:
        logg.info("\noutputs model fit of gene:", dm.gene)

    return dm if return_model else adata if copy else None


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

    This is the original scVelo implementation that uses compute_shared_time,
    root_time, and velocity pseudotime integration.
    """
    # Call the original implementation
    return latent_time_original(
        data=data,
        vkey=vkey,
        min_likelihood=min_likelihood,
        min_confidence=min_confidence,
        min_corr_diffusion=min_corr_diffusion,
        weight_diffusion=weight_diffusion,
        root_key=root_key,
        end_key=end_key,
        t_max=t_max,
        copy=copy,
    )


# ---------------------------------------------------------------------------
# Original scVelo latent_time implementation
# ---------------------------------------------------------------------------


def latent_time_original(
    data,
    vkey="velocity",
    min_likelihood=0.1,
    min_confidence=0.75,
    min_corr_diffusion=None,
    weight_diffusion=None,
    root_key=None,
    end_key=None,
    t_max=None,
    copy=False,
):
    """Computes a gene-shared latent time (original scVelo implementation).

    Gene-specific latent timepoints obtained from the dynamical model are coupled to a
    universal gene-shared latent time, which represents the cell's internal clock and
    is based only on its transcriptional dynamics.

    Parameters
    ----------
    data: :class:`~anndata.AnnData`
        Annotated data matrix
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    min_likelihood: `float` between `0` and `1` or `None` (default: `.1`)
        Minimal likelihood fitness for genes to be included to the weighting.
    min_confidence: `float` between `0` and `1` (default: `.75`)
        Parameter for local coherence selection.
    min_corr_diffusion: `float` between `0` and `1` or `None` (default: `None`)
        Only select genes that correlate with velocity pseudotime obtained
        from diffusion random walk on velocity graph.
    weight_diffusion: `float` or `None` (default: `None`)
        Weight applied to couple latent time with diffusion-based velocity pseudotime.
    root_key: `str` or `None` (default: `'root_cells'`)
        Key (.uns, .obs) of root cell to be used.
    end_key: `str` or `None` (default: `None`)
        Key (.obs) of end points to be used.
    t_max: `float` or `None` (default: `None`)
        Overall duration of differentiation process.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    latent_time: `.obs`
        latent time from learned dynamics for each cell
    """
    from ._em_model_utils import compute_shared_time, root_time
    from .utils import scale, vcorrcoef
    
    adata = data.copy() if copy else data

    if "fit_t" not in adata.layers.keys():
        raise ValueError("you need to run `tl.recover_dynamics` first.")

    if f"{vkey}_graph" not in adata.uns.keys():
        from scvelo.tools import velocity_graph
        velocity_graph(adata, approx=True)

    if root_key is None:
        terminal_keys = ["root_cells", "starting_cells", "root_states_probs"]
        keys = [key for key in terminal_keys if key in adata.obs.keys()]
        if len(keys) > 0:
            root_key = keys[0]
    if root_key not in adata.uns.keys() and root_key not in adata.obs.keys():
        root_key = "root_cells"
    if root_key not in adata.obs.keys():
        from scvelo.tools import terminal_states
        terminal_states(adata, vkey=vkey)

    t = np.array(adata.layers["fit_t"])
    idx_valid = ~np.isnan(t.sum(0))
    if min_likelihood is not None:
        likelihood = adata.var["fit_likelihood"].values
        idx_valid &= np.array(likelihood >= min_likelihood, dtype=bool)
    t = t[:, idx_valid]
    t_sum = np.sum(t, 1)
    conn = get_connectivities(adata)

    if root_key not in adata.uns.keys():
        roots = np.argsort(t_sum)
        idx_roots = np.array(adata.obs[root_key][roots])
        idx_roots[pd.isnull(idx_roots)] = 0
        if np.any([isinstance(ix, str) for ix in idx_roots]):
            idx_roots = np.array([isinstance(ix, str) for ix in idx_roots], dtype=int)
        idx_roots = idx_roots.astype(float) > 1 - 1e-3
        if np.sum(idx_roots) > 0:
            roots = roots[idx_roots]
        else:
            logg.warn(
                "No root cells detected. Consider specifying "
                "root cells to improve latent time prediction."
            )
    else:
        roots = [adata.uns[root_key]]
        root_key = f"root cell {adata.uns[root_key]}"

    if end_key in adata.obs.keys():
        fates = np.argsort(t_sum)[::-1]
        idx_fates = np.array(adata.obs[end_key][fates])
        idx_fates[pd.isnull(idx_fates)] = 0
        if np.any([isinstance(ix, str) for ix in idx_fates]):
            idx_fates = np.array([isinstance(ix, str) for ix in idx_fates], dtype=int)
        idx_fates = idx_fates.astype(float) > 1 - 1e-3
        if np.sum(idx_fates) > 0:
            fates = fates[idx_fates]
    else:
        fates = [None]

    logg.info(
        f"computing latent time using {root_key}"
        f"{', ' + end_key if end_key in adata.obs.keys() else ''} as prior",
        r=True,
    )

    from scvelo.tools import velocity_pseudotime
    VPT = velocity_pseudotime(
        adata, vkey, root_key=roots[0], end_key=fates[0], return_model=True
    )
    vpt = VPT.pseudotime

    if min_corr_diffusion is not None:
        corr = vcorrcoef(t.T, vpt)
        t = t[:, np.array(corr >= min_corr_diffusion, dtype=bool)]

    if root_key in adata.uns.keys():
        root = adata.uns[root_key]
        t, t_ = root_time(t, root=root)
        latent_time = compute_shared_time(t)
    else:
        roots = roots[:4]
        latent_time = np.ones(shape=(len(roots), adata.n_obs))
        for i, root in enumerate(roots):
            t, t_ = root_time(t, root=root)
            latent_time[i] = compute_shared_time(t)
        latent_time = scale(np.mean(latent_time, axis=0))

    if fates[0] is not None:
        fates = fates[:4]
        latent_time_ = np.ones(shape=(len(fates), adata.n_obs))
        for i, fate in enumerate(fates):
            t, t_ = root_time(t, root=fate)
            latent_time_[i] = 1 - compute_shared_time(t)
        latent_time = scale(latent_time + 0.2 * scale(np.mean(latent_time_, axis=0)))

    tl = latent_time
    tc = conn.dot(latent_time)

    z = tl.dot(tc) / tc.dot(tc)
    tl_conf = (1 - np.abs(tl / np.max(tl) - tc * z / np.max(tl))) ** 2
    idx_low_confidence = tl_conf < min_confidence

    if weight_diffusion is not None:
        w = weight_diffusion
        latent_time = (1 - w) * latent_time + w * vpt
        latent_time[idx_low_confidence] = vpt[idx_low_confidence]
    else:
        conn_new = conn.copy()
        conn_new[:, idx_low_confidence] = 0
        conn_new.eliminate_zeros()
        latent_time = conn_new.dot(latent_time)

    latent_time = scale(latent_time)
    if t_max is not None:
        latent_time *= t_max

    adata.obs["latent_time"] = latent_time

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    'latent_time', shared time (adata.obs)")
    return adata if copy else None


