from scvelo.preprocessing import filter_and_normalize as scv_filter_and_normalize
from scanpy.preprocessing import neighbors as sc_neighbors
from scvelo.preprocessing import moments as scv_moments
from scanpy.preprocessing import pca as sc_pca
from anndata import AnnData
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray


def filter_and_normalize(
    data: AnnData,
    min_counts: int | None = None,
    min_counts_u: int | None = None,
    min_cells: int | None = None,
    min_cells_u: int | None = None,
    min_shared_counts: int | None = None,
    min_shared_cells: int | None = None,
    n_top_genes: int | None = None,
    retain_genes: list | None = None,
    subset_highly_variable: bool = True,
    flavor: Literal["seurat", "cell_ranger", "svr"] = "seurat",
    log: bool = True,
    layers_normalize: list | None = None,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData | None:
    return scv_filter_and_normalize(
        data=data,
        min_counts=min_counts,
        min_counts_u=min_counts_u,
        min_cells=min_cells,
        min_cells_u=min_cells_u,
        min_shared_counts=min_shared_counts,
        min_shared_cells=min_shared_cells,
        n_top_genes=n_top_genes,
        retain_genes=retain_genes,
        subset_highly_variable=subset_highly_variable,
        flavor=flavor,
        log=log,
        layers_normalize=layers_normalize,
        copy=copy,
        **kwargs,
    )


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    knn: bool = True,
    method: Literal["umap", "gauss"] = "umap",
    transformer: Any = None,
    metric: Literal["euclidean"] = "euclidean",
    metric_kwds: dict[str, Any] = {},
    random_state: int = 0,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    return sc_neighbors(
        adata=adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        knn=knn,
        method=method,
        transformer=transformer,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
        key_added=key_added,
        copy=copy,
    )


def moments(
    data: AnnData,
    n_neighbors: int = 30,
    n_pcs: int | None = None,
    mode: Literal["connectivities", "distances"] = "connectivities",
    method: Literal["umap", "hnsw", "sklearn"] | None = "umap",
    use_rep: str | None = None,
    use_highly_variable: bool = True,
    copy: bool = False,
) -> AnnData | None:
    return scv_moments(
        data=data,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        mode=mode,
        method=method,  # type: ignore
        use_rep=use_rep,
        use_highly_variable=use_highly_variable,
        copy=copy,
    )


def pca(
    data: AnnData | NDArray | Any,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    zero_center: bool = True,
    svd_solver: Literal["arpack", "randomized", "auto", "covariance_eigh", "tsqr"]
    | None = None,
    chunked: bool = False,
    chunk_size: int | None = None,
    random_state: int = 0,
    return_info: bool = False,
    mask_var: NDArray[np.bool_] | str | None = None,
    use_highly_variable: bool | None = None,
    dtype: str = "float32",
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | NDArray | Any | None:
    return sc_pca(
        data=data,
        n_comps=n_comps,
        layer=layer,
        zero_center=zero_center,
        svd_solver=svd_solver,
        chunked=chunked,
        chunk_size=chunk_size,
        random_state=random_state,
        return_info=return_info,
        mask_var=mask_var,  # type: ignore
        use_highly_variable=use_highly_variable,
        dtype=dtype,
        key_added=key_added,
        copy=copy,
    )
