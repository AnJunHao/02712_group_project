# type: ignore
from typing import Any, Literal

import numpy as np
from anndata import AnnData
from scipy import sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

ArrayLike = np.ndarray | sp.spmatrix


def pca(
    data: AnnData,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    use_highly_variable: bool | None = None,
    zero_center: bool = True,
    dtype: np.dtype | str = "float32",
    key_added: str | None = None,
    copy: bool = False,
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "arpack",
    random_state: int | None = 0,
) -> AnnData:
    """
    Simplified PCA similar to scanpy.tl.pca, using only AnnData, numpy,
    scipy.sparse, and scikit-learn.

    Parameters
    ----------
    data
        AnnData or 2D array-like of shape (n_obs, n_vars).
    n_comps
        Number of principal components. If None, uses
        min(50, min(n_obs, n_vars) - 1).
    layer
        If `data` is AnnData, use this layer instead of `.X`.
    use_highly_variable
        If `data` is AnnData and `.var['highly_variable']` exists:
        - True: use only highly variable genes.
        - False: use all genes.
        - None: use highly variable genes if the column exists, else all genes.
    zero_center
        If True, run PCA (with centering) via `sklearn.decomposition.PCA`.
        If False, run truncated SVD via `sklearn.decomposition.TruncatedSVD`
        (no centering, works well with sparse data).
    dtype
        Data type of the returned representation (and stored results in AnnData).
    key_added
        If `data` is AnnData:
        - embedding stored in `adata.obsm[key_added]`
        - loadings stored in `adata.varm[key_added]`
        - stats stored in `adata.uns[key_added]`
        If None, use `'X_pca'`, `'PCs'`, `'pca'` like Scanpy.
    copy
        If `data` is AnnData:
        - True: return a new AnnData object with results.
        - False: modify in place and return the same object.
    svd_solver
        Passed to `sklearn.decomposition.PCA` when `zero_center=True`.
        For large problems, `'randomized'` can be much faster.
    random_state
        Random seed for randomized algorithms.

    Returns
    -------
    If `data` is AnnData:
        AnnData (same object if `copy=False`, otherwise a copy).
        Adds:

        - `.obsm['X_pca' | key_added]` : (n_obs, n_comps)
        - `.varm['PCs'   | key_added]` : (n_vars, n_comps)
        - `.uns['pca'    | key_added]` with keys:
            - 'variance'
            - 'variance_ratio'
            - 'params'

    If `data` is array-like:
        PCA scores as `np.ndarray` of shape (n_obs, n_comps).
    """
    # ------------------------------------------------------------------
    # 1) Normalize inputs
    # ------------------------------------------------------------------
    adata = data.copy() if copy else data

    # Choose matrix: X or layer
    if layer is None:
        X: ArrayLike = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers")
        X = adata.layers[layer]

    if not (isinstance(X, np.ndarray) or sp.isspmatrix(X)):
        raise TypeError(
            f"adata matrix must be numpy array or scipy.sparse.spmatrix, got {type(X)}"
        )

    # Handle highly variable gene selection
    mask_var: np.ndarray | None = None
    if use_highly_variable is not False:
        if "highly_variable" in adata.var.columns:
            if use_highly_variable is None or use_highly_variable is True:
                hv = adata.var["highly_variable"].to_numpy()
                if hv.dtype != bool:
                    hv = hv.astype(bool)
                mask_var = hv
        elif use_highly_variable is True:
            raise KeyError(
                "use_highly_variable=True, but adata.var['highly_variable'] "
                "does not exist."
            )

    if mask_var is not None:
        if mask_var.shape[0] != adata.n_vars:
            raise ValueError(
                "mask_var (highly_variable) length does not match adata.n_vars"
            )
        X_use = X[:, mask_var]
    else:
        X_use = X

    # ------------------------------------------------------------------
    # 2) Determine n_comps
    # ------------------------------------------------------------------
    n_obs, n_vars = X_use.shape
    if n_comps is None:
        max_comps = min(n_obs, n_vars) - 1 if zero_center else min(n_obs, n_vars)
        if max_comps < 1:
            raise ValueError(
                f"Cannot compute PCA with shape {X_use.shape}; "
                "need at least 2 observations and 2 variables."
            )
        n_comps = min(50, max_comps)
    else:
        if n_comps < 1:
            raise ValueError("n_comps must be >= 1")
        max_comps = min(n_obs, n_vars) - (1 if zero_center else 0)
        if n_comps > max_comps:
            raise ValueError(
                f"n_comps={n_comps} is too large for data shape {X_use.shape}. "
                f"Maximum allowed is {max_comps} when zero_center={zero_center}."
            )

    # ------------------------------------------------------------------
    # 3) Run PCA / TruncatedSVD
    # ------------------------------------------------------------------
    is_sparse = sp.isspmatrix(X_use)

    if zero_center:
        # For simplicity and robustness, densify sparse input.
        if is_sparse:
            X_prep = X_use.toarray().astype(dtype, copy=False)
        else:
            X_prep = np.asarray(X_use, dtype=dtype)

        pca_model = PCA(
            n_components=n_comps,
            svd_solver=svd_solver,
            random_state=random_state,
        )
        scores = pca_model.fit_transform(X_prep)
        loadings = pca_model.components_.T  # (n_features, n_comps)
        variance = pca_model.explained_variance_
        variance_ratio = pca_model.explained_variance_ratio_

    else:
        # TruncatedSVD can handle sparse directly
        if is_sparse:
            X_prep = X_use.tocsr()
        else:
            X_prep = np.asarray(X_use, dtype=dtype)

        svd = TruncatedSVD(
            n_components=n_comps,
            random_state=random_state,
        )
        scores = svd.fit_transform(X_prep)
        loadings = svd.components_.T  # (n_features, n_comps)
        variance = svd.explained_variance_
        variance_ratio = svd.explained_variance_ratio_

    # Ensure output dtype
    scores = np.asarray(scores, dtype=dtype)
    loadings = np.asarray(loadings, dtype=dtype)
    variance = np.asarray(variance, dtype=dtype)
    variance_ratio = np.asarray(variance_ratio, dtype=dtype)

    # ------------------------------------------------------------------
    # 4) Return or store in AnnData
    # ------------------------------------------------------------------
    # AnnData output
    key_obsm, key_varm, key_uns = (
        ("X_pca", "PCs", "pca")
        if key_added is None
        else (key_added, key_added, key_added)
    )

    # Store scores
    adata.obsm[key_obsm] = scores

    # Store loadings, padded to all genes if we used a mask
    n_vars_full = adata.n_vars
    pcs_full = np.zeros((n_vars_full, n_comps), dtype=dtype)
    if mask_var is not None:
        pcs_full[mask_var, :] = loadings
    else:
        pcs_full[:, :] = loadings
    adata.varm[key_varm] = pcs_full

    # Store variance info
    params: dict[str, Any] = {
        "zero_center": zero_center,
        "use_highly_variable": bool(mask_var is not None),
    }
    if layer is not None:
        params["layer"] = layer

    adata.uns[key_uns] = {
        "params": params,
        "variance": variance,
        "variance_ratio": variance_ratio,
    }

    return adata
