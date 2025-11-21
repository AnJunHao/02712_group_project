# type: ignore
"""
Simplified latent time estimation built on top of the recovered dynamics.

The goal of this module is not to reproduce every nuance of scVelo's latent
time implementation but to provide a deterministic, dependency-light variant
that plays nicely with the rest of our project.  We aggregate the gene-specific
latent times written by ``recover_dynamics`` and enforce neighborhood
consistency using the precomputed connectivities graph.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from project.preprocessing.moments import _get_connectivities

Array = np.ndarray


def _scale(values: Array) -> Array:
    """Min-max scale a vector while guarding against degenerate ranges."""

    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max == v_min:
        return np.zeros_like(values, dtype=np.float64)
    scaled = (values - v_min) / (v_max - v_min)
    return scaled


def _parse_prior_indices(adata: AnnData, key: str | None) -> np.ndarray | None:
    """
    Convert a categorical/boolean column into indices of prior cells.

    Any value that evaluates to ``True`` (for boolean) or ``> 0`` (for numeric)
    is considered part of the prior set.
    """

    if key is None or key not in adata.obs.columns:
        return None

    col = adata.obs[key]
    if col.dtype.kind in {"b"}:
        mask = col.to_numpy()
    elif col.dtype.kind in {"i", "u", "f"}:
        mask = np.asarray(col.to_numpy(), dtype=float) > 0
    else:
        # treat strings/categories as boolean via non-null check
        mask = ~col.isna().to_numpy()
    idx = np.where(mask)[0]
    return idx if idx.size > 0 else None


def infer_latent_time(
    adata: AnnData,
    gene_mask: Array,
    *,
    min_confidence: float = 0.75,
    min_likelihood: float = 0.1,
    weight_diffusion: float | None = None,
    root_key: str | None = None,
    end_key: str | None = None,
    t_max: float | None = None,
) -> Array:
    """
    Aggregate gene-specific latent times into a shared latent time.

    Parameters
    ----------
    adata
        Annotated data matrix with ``fit_t``.
    gene_mask
        Boolean mask indicating genes that pass the velocity filtering.
    min_confidence
        Threshold that determines whether a cell trusts its own time estimate
        or replaces it with the neighborhood average.
    min_likelihood
        Minimum gene likelihood (used for weighting). The argument mirrors the
        one from the public API so we keep a consistent signature.
    weight_diffusion
        If provided, interpolate between the raw latent time and the
        connectivity-smoothed version using ``weight_diffusion`` as weight.
    root_key, end_key
        Keys in ``adata.obs`` that label start/end cells.
    t_max
        If given, scale the latent time interval to ``[0, t_max]``.

    Returns
    -------
    np.ndarray
        Dense vector with per-cell latent time.
    """

    if "fit_t" not in adata.layers:
        raise ValueError("recover_dynamics has to run before latent time estimation.")

    latent = np.asarray(adata.layers["fit_t"][:, gene_mask], dtype=np.float64)

    if root_key is None:
        for candidate in ("root_cells", "starting_cells", "root_states_probs"):
            if candidate in adata.obs.columns:
                root_key = candidate
                break
    if end_key is None and "end_points" in adata.obs.columns:
        end_key = "end_points"
    likelihood = (
        np.asarray(adata.var["fit_likelihood"].values[gene_mask], dtype=np.float64)
        if "fit_likelihood" in adata.var.columns
        else np.ones(latent.shape[1], dtype=np.float64)
    )
    likelihood = np.clip(likelihood, 1e-8, None)

    valid = np.isfinite(latent)
    weights = valid * likelihood
    summed_weights = np.maximum(weights.sum(axis=1), 1e-8)
    weighted_latent = np.where(valid, latent, 0.0) * likelihood
    shared_time = weighted_latent.sum(axis=1) / summed_weights

    conn = _get_connectivities(adata)
    neighbor_time = conn.dot(shared_time)

    diff = np.abs(shared_time - neighbor_time)
    if diff.max() == 0:
        confidence = np.ones_like(diff)
    else:
        confidence = 1.0 - diff / diff.max()

    low_conf = confidence < min_confidence

    if weight_diffusion is not None:
        shared_time = (1.0 - weight_diffusion) * shared_time + weight_diffusion * neighbor_time
        shared_time[low_conf] = neighbor_time[low_conf]
    else:
        shared_time[low_conf] = neighbor_time[low_conf]

    shared_time = _scale(shared_time)

    roots = _parse_prior_indices(adata, root_key)
    if roots is not None and roots.size > 0:
        shared_time -= np.nanmean(shared_time[roots])
        shared_time = np.clip(shared_time, 0.0, None)

    ends = _parse_prior_indices(adata, end_key)
    if ends is not None and ends.size > 0:
        max_end = np.nanmean(shared_time[ends])
        if max_end > 0:
            shared_time /= max_end

    shared_time = _scale(shared_time)
    if t_max is not None:
        shared_time *= t_max

    return shared_time
