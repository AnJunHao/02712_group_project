# type: ignore
"""
Lightweight velocity-graph construction.

scVelo's visualization stack expects cosine correlations between cell-to-cell
transitions.  This module provides a simplified version that uses the neighbor
graph from :mod:`project.preprocessing` and computes correlations between the
velocity vector of a source cell and the difference vector pointing to each of
its neighbors.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from scipy import sparse as sp

from project.preprocessing.moments import _get_connectivities

Array = NDArray[np.float64]


def _get_expression(adata: AnnData, xkey: str) -> Array:
    """
    Fetch a dense expression matrix used for spatial displacements.

    Falls back to ``adata.X`` if the requested layer is absent.
    """

    if xkey == "X":
        X = adata.X
    else:
        X = adata.layers.get(xkey, None)
        if X is None and xkey == "Ms":
            # ``moments`` might not have been called; fall back to spliced counts
            X = adata.layers.get("spliced", None)
    if X is None:
        X = adata.X

    if sp.isspmatrix(X):
        return X.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _get_velocity(adata: AnnData, vkey: str) -> Array:
    """Return velocities as dense array."""

    if vkey not in adata.layers:
        raise KeyError(f"{vkey!r} not found in adata.layers.")
    V = adata.layers[vkey]
    if sp.isspmatrix(V):
        return V.toarray().astype(np.float64, copy=False)
    return np.asarray(V, dtype=np.float64)


def velocity_graph(
    data: AnnData,
    vkey: str = "velocity",
    xkey: str = "Ms",
    copy: bool = False,
) -> AnnData | None:
    """
    Compute cosine correlations between RNA velocities and neighbor displacements.

    The resulting sparse matrices are stored in ``adata.uns[f\"{vkey}_graph\"]`` and
    ``adata.uns[f\"{vkey}_graph_neg\"]`` to match scVelo's public API.
    """

    adata = data.copy() if copy else data

    conn = _get_connectivities(adata)
    if conn.nnz == 0:
        raise ValueError(
            "Neighbor graph is empty. Run project.pp.neighbors / moments first."
        )

    X = _get_expression(adata, xkey)
    V = _get_velocity(adata, vkey)

    rows, cols = conn.nonzero()
    n_edges = rows.size
    values = np.zeros(n_edges, dtype=np.float64)

    for idx, (i, j) in enumerate(zip(rows, cols)):
        delta = X[j] - X[i]
        vel = V[i]
        mask = np.isfinite(delta) & np.isfinite(vel)
        if not np.any(mask):
            continue
        delta = delta[mask]
        vel = vel[mask]
        denom = np.linalg.norm(delta) * np.linalg.norm(vel)
        if denom > 0:
            values[idx] = float(np.dot(delta, vel) / denom)

    graph = sp.csr_matrix((values, (rows, cols)), shape=conn.shape, dtype=np.float32)
    graph_neg = graph.copy()

    # Split positive and negative correlations
    graph.data = np.clip(graph.data, 0, None)
    graph_neg.data = np.clip(graph_neg.data, None, 0)
    graph.eliminate_zeros()
    graph_neg.eliminate_zeros()

    adata.uns[f"{vkey}_graph"] = graph
    adata.uns[f"{vkey}_graph_neg"] = graph_neg

    return adata if copy else None

