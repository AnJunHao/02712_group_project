# type: ignore
from typing import Any

import numpy as np
from anndata import AnnData
from scipy import sparse as sp
from sklearn.neighbors import NearestNeighbors


def _choose_representation(
    adata: AnnData,
    use_rep: str | None = None,
    n_pcs: int | None = None,
) -> np.ndarray | sp.spmatrix:
    """
    Simplified representation selection similar to Scanpy:

    Priority:
    1. If use_rep is not None:
       - "X" → adata.X
       - otherwise → adata.obsm[use_rep]
    2. Else if n_pcs is not None and 'X_pca' in adata.obsm:
       → adata.obsm['X_pca'][:, :n_pcs]
    3. Else:
       → adata.X
    """
    if use_rep is not None:
        if use_rep == "X":
            X = adata.X
        elif use_rep in adata.obsm:
            X = adata.obsm[use_rep]
        else:
            raise KeyError(
                f"use_rep={use_rep!r} not found in adata.obsm or as 'X'. "
                "Available .obsm keys: "
                f"{list(adata.obsm.keys())}"
            )
    elif n_pcs is not None and "X_pca" in adata.obsm:
        X = adata.obsm["X_pca"][:, :n_pcs]
    else:
        X = adata.X

    if not (isinstance(X, np.ndarray) or sp.isspmatrix(X)):
        raise TypeError(
            f"Chosen representation must be numpy array or scipy.sparse.spmatrix, "
            f"got {type(X)}"
        )
    return X


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    metric: str = "euclidean",
    key_added: str | None = None,
    copy: bool = False,
    random_state: int
    | None = None,  # kept for API similarity, not used by NearestNeighbors
) -> AnnData | None:
    """
    Compute a k‑nearest neighbors graph and distances, Scanpy‑style, with minimal
    dependencies and a simplified backend.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        Number of nearest neighbors (k) to use.
    n_pcs
        If not None and ``'X_pca'`` exists in ``adata.obsm``, use the first
        `n_pcs` principal components as input.
    use_rep
        Which representation to use:

        - ``None`` (default): use `X_pca[:, :n_pcs]` if possible, else `.X`.
        - ``"X"``: use `adata.X`.
        - any other string: use `adata.obsm[use_rep]`.
    metric
        Distance metric passed to `sklearn.neighbors.NearestNeighbors`, e.g.
        `"euclidean"`, `"cosine"`, `"manhattan"`, ...
    key_added
        - If None (default): store in
          - `.uns['neighbors']`
          - `.obsp['distances']`
          - `.obsp['connectivities']`
        - Else: store in
          - `.uns[key_added]`
          - `.obsp[key_added + '_distances']`
          - `.obsp[key_added + '_connectivities']`
    copy
        If True, return a copy of `adata` with neighbors stored.
        If False, modify `adata` in-place and return None.
    random_state
        Present for API similarity; not used by this simplified implementation.

    Returns
    -------
    If `copy=False` (default): returns None and modifies `adata` in-place.

    If `copy=True`: returns a new `AnnData` with:

    - `adata.obsp['distances' | key_added+'_distances']`
        CSR sparse matrix of pairwise distances to k neighbors (excluding self).
    - `adata.obsp['connectivities' | key_added+'_connectivities']`
        Symmetric kNN connectivity graph (CSR, float32).
    - `adata.uns['neighbors' | key_added]`
        Dictionary with parameters and keys for the above matrices.

    Notes
    -----
    - Uses exact kNN via `sklearn.neighbors.NearestNeighbors`.
    - Connectivities are a simple symmetric, distance‑weighted kNN graph,
      **not** the full UMAP affinity construction used by Scanpy internally.
    """
    if adata.n_obs < 2:
        raise ValueError("Need at least 2 observations to compute neighbors.")

    adata = adata.copy() if copy else adata

    # ------------------------------------------------------------------
    # 1) Choose representation X
    # ------------------------------------------------------------------
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    # For simplicity and maximum metric support, densify sparse matrices
    if sp.isspmatrix(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    n_obs = X_dense.shape[0]
    if n_neighbors >= n_obs:
        # Mirror Scanpy behavior: cap at n_obs - 1
        n_neighbors = n_obs - 1

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1 and < n_obs.")

    # ------------------------------------------------------------------
    # 2) Fit exact kNN using sklearn
    # ------------------------------------------------------------------
    # We ask for n_neighbors+1 and drop the self‑neighbor
    n_neighbors_query = n_neighbors + 1

    nn = NearestNeighbors(
        n_neighbors=n_neighbors_query,
        metric=metric,
        algorithm="auto",
    )
    nn.fit(X_dense)
    dist_full, ind_full = nn.kneighbors(X_dense, return_distance=True)

    # ------------------------------------------------------------------
    # 3) Remove self‑neighbors and build CSR distance matrix
    # ------------------------------------------------------------------
    rows = []
    cols = []
    data = []

    for i in range(n_obs):
        inds_i = ind_full[i]
        dists_i = dist_full[i]

        # Remove self index (if present) and keep up to n_neighbors others
        mask = inds_i != i
        inds_i = inds_i[mask]
        dists_i = dists_i[mask]

        if inds_i.size > n_neighbors:
            inds_i = inds_i[:n_neighbors]
            dists_i = dists_i[:n_neighbors]

        # Record for CSR
        rows.extend([i] * len(inds_i))
        cols.extend(inds_i.tolist())
        data.extend(dists_i.tolist())

    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    data = np.array(data, dtype=np.float32)

    distances = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))

    # ------------------------------------------------------------------
    # 4) Build connectivities as symmetric, distance‑weighted kNN graph
    # ------------------------------------------------------------------
    # Simple similarity: w_ij = exp(-d_ij / mean_distance)
    if data.size == 0:
        # Degenerate case, no edges
        connectivities = sp.csr_matrix(distances.shape, dtype=np.float32)
    else:
        mean_dist = float(np.mean(data))
        scale = mean_dist + 1e-8
        conn_data = np.exp(-data / scale).astype(np.float32)
        connectivities = sp.csr_matrix((conn_data, (rows, cols)), shape=(n_obs, n_obs))
        # Symmetrize: undirected graph
        connectivities = connectivities.maximum(connectivities.T)

    # ------------------------------------------------------------------
    # 5) Store results in AnnData
    # ------------------------------------------------------------------
    if key_added is None:
        neighbors_key = "neighbors"
        dists_key = "distances"
        conns_key = "connectivities"
    else:
        neighbors_key = key_added
        dists_key = f"{key_added}_distances"
        conns_key = f"{key_added}_connectivities"

    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    params: dict[str, Any] = {
        "n_neighbors": n_neighbors,
        "metric": metric,
    }
    if use_rep is not None:
        params["use_rep"] = use_rep
    if n_pcs is not None:
        params["n_pcs"] = n_pcs

    adata.uns[neighbors_key] = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": params,
    }

    return adata if copy else None
