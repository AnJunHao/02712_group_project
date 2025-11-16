# type: ignore
from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse

from project.preprocessing.neighbors import neighbors


def _ensure_neighbors(
    adata: AnnData,
    n_neighbors: int,
    n_pcs: Optional[int],
    use_rep: Optional[str],
) -> None:
    """
    Ensure that a neighbors graph exists in `adata` with at least `n_neighbors`.

    This is designed to work with:
    - The simplified `neighbors()` implementation discussed earlier, or
    - A standard Scanpy `pp.neighbors` graph (same uns/obsp layout).
    """
    if n_neighbors is None:
        # Fall back to whatever is already in `adata.uns['neighbors']`
        if "neighbors" not in adata.uns:
            raise ValueError(
                "No neighbor graph found in `adata.uns['neighbors']` and "
                "`n_neighbors` is None. Please provide `n_neighbors` or run "
                "`neighbors()` / `scanpy.pp.neighbors()` first."
            )
        return

    # If no neighbors yet, compute
    if "neighbors" not in adata.uns:
        neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
        )
        return

    # If neighbors exist, check whether they are sufficient
    info = adata.uns.get("neighbors", {})
    params = info.get("params", {})
    existing_n_neighbors = params.get("n_neighbors", None)
    existing_n_pcs = params.get("n_pcs", None)
    existing_use_rep = params.get("use_rep", None)

    need_recompute = False

    # If existing graph has fewer neighbors than requested, recompute
    if existing_n_neighbors is None or existing_n_neighbors < n_neighbors:
        need_recompute = True

    # If caller explicitly requested a representation that doesn't match, recompute
    if (
        use_rep is not None
        and existing_use_rep is not None
        and use_rep != existing_use_rep
    ):
        need_recompute = True

    # If caller explicitly requested a different n_pcs, recompute
    if n_pcs is not None and existing_n_pcs is not None and n_pcs != existing_n_pcs:
        need_recompute = True

    if need_recompute:
        neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
        )


def _get_connectivities(
    adata: AnnData,
) -> csr_matrix:
    """
    Retrieve the neighbor connectivities matrix as CSR.

    Compatible with:
    - The simplified `neighbors()` implementation (stores under `.uns['neighbors']`
      and `.obsp['connectivities']`), and
    - Standard Scanpy `pp.neighbors` output.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "No neighbor information found in `adata.uns['neighbors']`. "
            "Run `neighbors()` / `scanpy.pp.neighbors()` first."
        )

    neighbors_info = adata.uns.get("neighbors", {})
    conn_key = neighbors_info.get("connectivities_key", "connectivities")

    if conn_key not in adata.obsp:
        raise KeyError(
            f"Connectivity matrix `{conn_key}` (from `adata.uns['neighbors']`) "
            f"not found in `adata.obsp`."
        )

    conn = adata.obsp[conn_key]
    if not issparse(conn):
        conn = csr_matrix(conn)
    else:
        conn = conn.tocsr()

    # Ensure float32 for downstream numerical stability / memory
    if conn.dtype != np.float32:
        conn = conn.astype(np.float32)

    return conn


def moments(
    data: AnnData,
    n_neighbors: int = 30,
    n_pcs: Optional[int] = None,
    mode: Literal["connectivities", "distances"] = "connectivities",
    method: str = "umap",
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Compute first-order moments for velocity estimation using a kNN graph.

    This is a **minimal reimplementation** of scVelo's `tl.moments` that only
    depends on:

    - `AnnData`
    - `numpy`
    - `scipy.sparse`

    and assumes a neighbors graph produced by either:

    - The simplified `neighbors()` function discussed earlier, or
    - `scanpy.pp.neighbors`.

    The main result is:

    - `adata.layers["Ms"]`: first-order moments of spliced counts
    - `adata.layers["Mu"]`: first-order moments of unspliced counts

    Parameters
    ----------
    data
        Annotated data matrix (`AnnData`).
    n_neighbors
        Number of neighbors to use for moment computation. If a neighbors graph
        already exists with fewer neighbors, it will be recomputed.
    n_pcs
        Number of principal components to use when computing neighbors (passed
        through to `neighbors()`). If `None`, the neighbors function decides
        (e.g. use `X_pca` if present, otherwise `.X`).
    mode
        Only `'connectivities'` is supported in this simplified implementation.
        (The original API also allows `'distances'`.)
    method
        Present for API similarity with scVelo. **Ignored** here; neighbors are
        always computed using the exact sklearn-based backend from the simplified
        `neighbors()` implementation.
    use_rep
        Which representation to use for computing neighbors (passed to
        `neighbors()`). Examples: `None`, `"X"`, `"X_pca"`, or any key in
        `adata.obsm`.
    use_highly_variable
        Present for API compatibility. Not used in this simplified version; we
        assume that any HVG filtering has already been applied upstream (e.g. in
        PCA computation).
    copy
        If `True`, return a copy of `data` with the computed moments.
        If `False`, operate in-place and return `None`.

    Returns
    -------
    If `copy=False` (default): returns `None` and writes results to `data` in-place.

    If `copy=True`: returns a new `AnnData` object with added layers:

    - `adata.layers["Ms"]` – dense `float32` array (n_cells × n_genes)
    - `adata.layers["Mu"]` – dense `float32` array (n_cells × n_genes)

    Notes
    -----
    - This function **does not** perform any automatic per-cell normalization.
      It operates on whatever values are stored in `adata.layers["spliced"]`
      and `adata.layers["unspliced"]` (raw counts, log-normalized, etc.).
    - Second-order moments (`Mss`, `Mus`, `Muu`) and generic `get_moments`
      are not implemented here; only the main `moments()` entry point.
    """
    # ------------------------------------------------------------------
    # 0) API simplifications / restrictions
    # ------------------------------------------------------------------
    if mode != "connectivities":
        raise NotImplementedError(
            "This simplified `moments` implementation currently only supports "
            "mode='connectivities'."
        )

    # `method` and `use_highly_variable` are accepted for API compatibility
    # but not used. We keep them to allow existing calling code to run without
    # modification. No behavior depends on them here.

    # ------------------------------------------------------------------
    # 1) Copy semantics
    # ------------------------------------------------------------------
    adata = data.copy() if copy else data

    # ------------------------------------------------------------------
    # 2) Check for required layers
    # ------------------------------------------------------------------
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise ValueError(
            "Both 'spliced' and 'unspliced' must be present in `adata.layers` "
            "to compute moments."
        )

    # ------------------------------------------------------------------
    # 3) Ensure neighbors graph exists (compute if needed)
    # ------------------------------------------------------------------
    _ensure_neighbors(
        adata=adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
    )

    # ------------------------------------------------------------------
    # 4) Get connectivities matrix
    # ------------------------------------------------------------------
    connectivities = _get_connectivities(adata)

    # ------------------------------------------------------------------
    # 5) Compute Ms and Mu (first-order moments)
    # ------------------------------------------------------------------
    # We convert layers to CSR for efficient multiplication. This works for
    # both dense and sparse layers.
    spliced = csr_matrix(adata.layers["spliced"])
    unspliced = csr_matrix(adata.layers["unspliced"])

    Ms = connectivities.dot(spliced).astype(np.float32).toarray()
    Mu = connectivities.dot(unspliced).astype(np.float32).toarray()

    adata.layers["Ms"] = Ms
    adata.layers["Mu"] = Mu

    return adata if copy else None
