

# type: ignore
from collections.abc import Sequence

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

ArrayLike = np.ndarray | sp.spmatrix


def filter_and_normalize(
    adata: AnnData,
    *,
    # gene filtering thresholds (on X)
    min_counts: int | None = None,
    min_cells: int | None = None,
    # gene filtering thresholds on unspliced layer (if present)
    min_counts_u: int | None = None,
    min_cells_u: int | None = None,
    # optional "shared" thresholds if both X/spliced and unspliced exist
    min_shared_counts: int | None = None,
    min_shared_cells: int | None = None,
    # highly variable genes
    n_top_genes: int | None = None,
    retain_genes: Sequence[str] | None = None,
    subset_highly_variable: bool = True,
    # log transform X after normalization
    log: bool = True,
    # normalization
    layers_normalize: str
    | Sequence[str]
    | None = None,  # None, "all", or list of layer names (without "X")
    counts_per_cell_after: float
    | None = None,  # target total per cell; if None, uses median
    key_n_counts: str = "n_counts",  # column in adata.obs to store new totals
    copy: bool = False,
) -> AnnData | None:
    """
    Reimplementation of a 'filter_and_normalize' style preprocessing function,
    using only AnnData, numpy, and scipy.sparse.

    Parameters
    ----------
    adata : AnnData
        AnnData object with count data in `.X` and optionally layers like
        "spliced" and "unspliced".
    min_counts : int or None
        Minimum total counts per gene (on adata.X) to retain it.
    min_cells : int or None
        Minimum number of cells expressing the gene (adata.X > 0) to retain it.
    min_counts_u : int or None
        Minimum total counts per gene on layer "unspliced" to retain.
    min_cells_u : int or None
        Minimum number of cells expressing gene on "unspliced" to retain.
    min_shared_counts : int or None
        Minimum total counts of gene in X + unspliced to retain (if both exist).
    min_shared_cells : int or None
        Minimum number of cells where gene is expressed in both X and unspliced.
    n_top_genes : int or None
        If not None, select this many highly variable genes.
    retain_genes : sequence of str or None
        Gene names (adata.var.index) that must always be retained.
    subset_highly_variable : bool
        If True, subset adata to highly variable genes; else only flag them.
    log : bool
        If True, apply log1p to adata.X after normalization.
    layers_normalize : None, "all", or sequence of str
        Which layers (besides X) to normalize per cell.
        - None: normalize X and any of ["spliced", "unspliced"] if present.
        - "all": normalize X and all layers.
        - sequence of str: normalize X and those layers.
    counts_per_cell_after : float or None
        Target total counts per cell; if None, use median per-cell counts.
    key_n_counts : str
        Column name to store (post-normalization) total counts per cell in adata.obs.
    copy : bool
        If True, operate on a copy and return it; if False, modify in place.

    Returns
    -------
    AnnData or None
        If copy=True, returns the processed AnnData, otherwise returns None.
    """
    if copy:
        adata = adata.copy()
    n_cells, n_genes = adata.shape

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _is_sparse(X: ArrayLike) -> bool:
        return sp.isspmatrix_csr(X) or sp.isspmatrix_csc(X) or sp.isspmatrix_coo(X)

    def _sum_per_gene(X: ArrayLike) -> np.ndarray:
        """Sum over cells (axis=0), dense or sparse."""
        if _is_sparse(X):
            return np.asarray(X.sum(axis=0)).ravel()
        else:
            return X.sum(axis=0)

    def _sum_per_cell(X: ArrayLike) -> np.ndarray:
        """Sum over genes (axis=1), dense or sparse."""
        if _is_sparse(X):
            return np.asarray(X.sum(axis=1)).ravel()
        else:
            return X.sum(axis=1)

    def _nnz_per_gene(X: ArrayLike) -> np.ndarray:
        """Number of cells where gene is expressed (X > 0)."""
        if _is_sparse(X):
            return np.asarray((X > 0).sum(axis=0)).ravel()
        else:
            return (X > 0).sum(axis=0)

    def _nnz_per_cell(X: ArrayLike) -> np.ndarray:
        """Number of genes expressed per cell (X > 0)."""
        if _is_sparse(X):
            return np.asarray((X > 0).sum(axis=1)).ravel()
        else:
            return (X > 0).sum(axis=1)

    def _subset_genes(mask: np.ndarray) -> None:
        """
        Subset adata to genes where mask is True, using AnnData's internal
        helper to keep X, layers, var, varm, raw, etc. consistent.
        """
        nonlocal adata, n_genes
        if mask.dtype != bool or mask.shape[0] != adata.n_vars:
            raise ValueError("mask must be boolean array of length n_genes")

        # This updates X, layers, var, varm, raw, etc. in one go.
        adata._inplace_subset_var(mask)
        n_genes = adata.n_vars

    def _ensure_float_matrix(X: ArrayLike) -> ArrayLike:
        """Ensure matrix is float32; works for dense or sparse."""
        if _is_sparse(X):
            if not np.issubdtype(X.dtype, np.floating):
                X = X.astype(np.float32)
        else:
            if not np.issubdtype(X.dtype, np.floating):
                X = X.astype(np.float32)
        return X

    def _row_scale_inplace(X: ArrayLike, scale_factors: np.ndarray) -> ArrayLike:
        """
        In-place row scaling: divide row i by scale_factors[i].
        scale_factors shape: (n_rows,)
        """
        sf = np.asarray(scale_factors).ravel()
        if _is_sparse(X):
            if not sp.isspmatrix_csr(X):
                X = X.tocsr()
            indptr = X.indptr
            data = X.data
            for i in range(X.shape[0]):
                start, end = indptr[i], indptr[i + 1]
                if end > start:
                    data[start:end] /= sf[i]
        else:
            X /= sf[:, None]
        return X

    def _get_layer(adata_obj: AnnData, key: str) -> ArrayLike | None:
        if key == "X":
            return adata_obj.X
        else:
            return adata_obj.layers.get(key, None)

    # ---------------------------------------------------------------------
    # 1) Gene filtering
    # ---------------------------------------------------------------------
    gene_keep: np.ndarray = np.ones(n_genes, dtype=bool)

    X_main: ArrayLike = adata.X
    # We'll treat adata.X as the main count layer for filtering.
    if min_counts is not None:
        sums = _sum_per_gene(X_main)
        gene_keep &= sums >= min_counts

    if min_cells is not None:
        nnz = _nnz_per_gene(X_main)
        gene_keep &= nnz >= min_cells

    # Use "unspliced" layer if present for _u thresholds
    unspliced: ArrayLike | None = adata.layers.get("unspliced", None)
    if unspliced is not None:
        if min_counts_u is not None:
            sums_u = _sum_per_gene(unspliced)
            gene_keep &= sums_u >= min_counts_u
        if min_cells_u is not None:
            nnz_u = _nnz_per_gene(unspliced)
            gene_keep &= nnz_u >= min_cells_u

        # "shared" between X and unspliced
        if min_shared_counts is not None or min_shared_cells is not None:
            sums_X = _sum_per_gene(X_main)
            sums_u = _sum_per_gene(unspliced)
            shared_counts = sums_X + sums_u

            shared_mask = np.ones(n_genes, dtype=bool)
            if min_shared_counts is not None:
                shared_mask &= shared_counts >= min_shared_counts

            if min_shared_cells is not None:
                # cells where both are expressed
                if _is_sparse(X_main):
                    both_expr = (X_main > 0).multiply(unspliced > 0)
                    shared_cells = np.asarray(both_expr.sum(axis=0)).ravel()
                else:
                    both_expr = (X_main > 0) & (unspliced > 0)
                    shared_cells = both_expr.sum(axis=0)
                shared_mask &= shared_cells >= min_shared_cells

            gene_keep &= shared_mask

    # Always retain specific genes if requested
    if retain_genes is not None:
        retain_set = set(retain_genes)
        idx_retain = np.array([g in retain_set for g in adata.var.index], dtype=bool)
        gene_keep |= idx_retain

    # Apply gene filtering if any gene is dropped
    if not np.all(gene_keep):
        _subset_genes(gene_keep)

    # ---------------------------------------------------------------------
    # 2) Per-cell normalization
    # ---------------------------------------------------------------------
    # Determine which layers to normalize
    if layers_normalize is None:
        # default: normalize X and any of ["spliced", "unspliced"] if present
        layer_names: list[str] = ["X"]
        for ln in ("spliced", "unspliced"):
            if ln in adata.layers:
                layer_names.append(ln)
    elif layers_normalize == "all":
        layer_names = ["X"] + list(adata.layers.keys())
    else:
        # user provided list of layer names; always include X
        if isinstance(layers_normalize, str):
            layers_normalize = [layers_normalize]
        layer_names = ["X"] + list(layers_normalize)

    # normalize each selected layer independently
    for lname in layer_names:
        layer = _get_layer(adata, lname)
        if layer is None:
            continue

        layer = _ensure_float_matrix(layer)
        # compute per-cell total counts for this layer
        counts = _sum_per_cell(layer)

        # decide target total per cell
        if counts_per_cell_after is None:
            nonzero = counts[counts > 0]
            if nonzero.size > 0:
                target = float(np.median(nonzero))
            else:
                target = 1.0
        else:
            target = float(counts_per_cell_after)

        # scaling factors (counts / target), avoid division by zero
        scale = counts / target
        scale[scale == 0] = 1.0

        # apply in-place row scaling
        layer = _row_scale_inplace(layer, scale)

        # write back
        if lname == "X":
            adata.X = layer
        else:
            adata.layers[lname] = layer

    # store total counts per cell from X after normalization
    adata.obs[key_n_counts] = _sum_per_cell(adata.X)

    # ---------------------------------------------------------------------
    # 3) Highly variable gene selection (Seurat-like)
    # ---------------------------------------------------------------------
    if n_top_genes is not None and n_top_genes < adata.n_vars:
        X_norm: ArrayLike = adata.X
        if _is_sparse(X_norm):
            # mean & variance for sparse
            mean = np.asarray(X_norm.mean(axis=0)).ravel()
            mean_sq = np.asarray(X_norm.multiply(X_norm).mean(axis=0)).ravel()
        else:
            mean = X_norm.mean(axis=0)
            mean_sq = (X_norm**2).mean(axis=0)
        var = mean_sq - mean**2

        # avoid numerical issues
        eps = 1e-12
        mean_clip = np.clip(mean, eps, None)
        dispersion = var / mean_clip

        # bin genes by mean
        n_bins = 20
        finite_mask = np.isfinite(mean) & np.isfinite(dispersion)
        mean_finite = mean[finite_mask]
        dispersion_finite = dispersion[finite_mask]

        if mean_finite.size < n_bins:
            n_bins = max(1, mean_finite.size)

        if n_bins > 1 and mean_finite.size > 0:
            # bin edges on quantiles of mean
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(mean_finite, quantiles)
            # digitize all genes
            bin_indices = np.digitize(mean, bin_edges[1:-1])
        else:
            bin_indices = np.zeros_like(mean, dtype=int)

        dispersion_norm = np.zeros_like(dispersion)

        for b in range(n_bins):
            bin_mask = (bin_indices == b) & finite_mask
            if not np.any(bin_mask):
                continue
            disp_bin = dispersion[bin_mask]
            mu = float(np.mean(disp_bin))
            std = float(np.std(disp_bin))
            if std == 0:
                std = 1.0
            dispersion_norm[bin_mask] = (dispersion[bin_mask] - mu) / std

        # pick top n_top_genes by normalized dispersion
        valid = np.isfinite(dispersion_norm)
        idx_valid = np.where(valid)[0]
        if idx_valid.size <= n_top_genes:
            hv_genes_idx = idx_valid
        else:
            order = np.argsort(dispersion_norm[valid])
            hv_genes_idx = idx_valid[order[-n_top_genes:]]

        highly_variable = np.zeros(adata.n_vars, dtype=bool)
        highly_variable[hv_genes_idx] = True

        # always retain specific genes if requested
        if retain_genes is not None:
            retain_set = set(retain_genes)
            idx_retain = np.array(
                [g in retain_set for g in adata.var.index], dtype=bool
            )
            highly_variable |= idx_retain

        # store statistics
        adata.var["means"] = mean
        adata.var["variances"] = var
        adata.var["dispersions"] = dispersion
        adata.var["dispersions_norm"] = dispersion_norm
        adata.var["highly_variable"] = highly_variable

        if subset_highly_variable:
            _subset_genes(highly_variable)

    # ---------------------------------------------------------------------
    # 4) Log-transform X
    # ---------------------------------------------------------------------
    if log:
        X_final: ArrayLike = adata.X
        if _is_sparse(X_final):
            X_csr = X_final.tocsr(copy=True)
            X_csr.data = np.log1p(X_csr.data)
            adata.X = X_csr
        else:
            adata.X = np.log1p(X_final)

    return adata if copy else None
