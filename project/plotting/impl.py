from graphvelo.plot import gene_score_histogram
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from tqdm import tqdm
import logging
import pandas as pd
import scipy.sparse as sp
import scvelo as scv

scv.settings.figdir = "temp"
scv.settings.set_figure_params(
    "scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis"
)
scv.settings.plot_prefix = ""

velocity_embedding_stream = scv.pl.velocity_embedding_stream
scatter = scv.pl.scatter


def flatten(arr: pd.Series | sp.csr_matrix | np.ndarray) -> np.ndarray:
    if type(arr) is pd.core.series.Series:  # type: ignore
        ret = arr.values.flatten()  # type: ignore
    elif sp.issparse(arr):
        ret = arr.A.flatten()  # type: ignore
    else:
        ret = arr.flatten()  # type: ignore
    return ret


def uniform_downsample_cells(X: np.ndarray, downsample):
    n_cells = X.shape[0]
    if 0 < downsample < 1:
        target = int(n_cells * downsample)
    else:
        target = int(downsample)

    n_bins = int(np.ceil(np.sqrt(target)))
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    x_bin = np.minimum(np.digitize(X[:, 0], x_edges) - 1, n_bins - 1)
    y_bin = np.minimum(np.digitize(X[:, 1], y_edges) - 1, n_bins - 1)
    bins = list(zip(x_bin, y_bin))

    selected_indices = []
    unique_bins = np.unique(bins, axis=0)

    for b in unique_bins:
        idxs = [i for i, bin_val in enumerate(bins) if bin_val == tuple(b)]
        if len(idxs) == 0:
            continue
        ix, iy = b
        x_center = (x_edges[ix] + x_edges[ix + 1]) / 2
        y_center = (y_edges[iy] + y_edges[iy + 1]) / 2
        distances = [
            np.sqrt((X[i, 0] - x_center) ** 2 + (X[i, 1] - y_center) ** 2) for i in idxs
        ]
        best_idx = idxs[np.argmin(distances)]
        selected_indices.append(best_idx)

    selected_indices = np.array(selected_indices)
    if selected_indices.size > target:
        selected_indices = np.random.choice(
            selected_indices, size=target, replace=False
        )

    return selected_indices


def plot_velocity_phase(
    adata,
    genes,
    s_layer: str = "M_s",
    u_layer: str = "M_u",
    vs_layer: str = "velocity_S",
    vu_layer: str = "velocity_U",
    smooth: bool = True,
    iteration: int = 5,
    beta: float = 0.1,
    color: str = "celltype",
    downsample: float | None = None,
    scale: float = 1.0,
    alpha: float = 0.4,
    quiver_alpha: float = 0.3,
    quiver_color: str = "black",
    show: bool = True,
    cmap="plasma",
    pointsize=1,
    figsize=None,
    ncols=3,
    dpi=100,
):
    if isinstance(genes, str):
        genes = [genes]

    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f"{missing_genes} not found")
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        raise ValueError("genes not found in adata.var_names")
    if gn < ncols:
        ncols = gn

    cell_annot = None
    if color in adata.obs and is_numeric_dtype(adata.obs[color]):
        colors = adata.obs[color].values
    elif (
        color in adata.obs
        and is_categorical_dtype(adata.obs[color])
        and color + "_colors" in adata.uns.keys()
    ):
        cell_annot = adata.obs[color].cat.categories
        if isinstance(adata.uns[f"{color}_colors"], dict):
            colors = list(adata.uns[f"{color}_colors"].values())
        elif isinstance(adata.uns[f"{color}_colors"], list):
            colors = adata.uns[f"{color}_colors"]
        elif isinstance(adata.uns[f"{color}_colors"], np.ndarray):
            colors = adata.uns[f"{color}_colors"].tolist()
        else:
            raise ValueError(f"Unsupported adata.uns[{color}_colors] object")
    else:
        raise ValueError(
            "Currently, color key must be a single string of "
            "either numerical or categorical available in adata"
            " obs, and the colors of categories can be found in"
            " adata uns."
        )

    nrows = -(-gn // ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(6 * ncols, 4 * (-(-gn // ncols))) if figsize is None else figsize,
        tight_layout=True,
        dpi=dpi,
    )

    axs = np.reshape(axs, (nrows, ncols))
    logging.info("Plotting trends")

    cnt = 0
    for cnt, gene in tqdm(
        enumerate(genes),
        total=gn,
        desc="Plotting velocity in phase diagram",
    ):
        spliced = flatten(adata[:, gene].layers[s_layer])
        unspliced = flatten(adata[:, gene].layers[u_layer])
        vs = flatten(adata[:, gene].layers[vs_layer])
        vu = flatten(adata[:, gene].layers[vu_layer])

        X_org = np.column_stack((spliced, unspliced))
        X = X_org.copy()
        V = np.column_stack((vs, vu))

        if smooth:
            nbrs_idx = adata.uns["neighbors"]["indices"]
            prev_score = V
            cur_score = np.zeros(prev_score.shape)
            for _ in range(iteration):
                for i in range(len(prev_score)):
                    vi = prev_score[nbrs_idx[i]]
                    cur_score[i] = (beta * vi[0]) + ((1 - beta) * vi[1:].mean(axis=0))
                prev_score = cur_score
            V = cur_score

        n_total = X.shape[0]
        if downsample is not None:
            if 0 < downsample <= 1:
                target_num = int(n_total * downsample)
            else:
                target_num = int(downsample)
            selected_idx = uniform_downsample_cells(X, target_num)
            X = X[selected_idx]
            V = V[selected_idx]

        row = cnt // ncols
        col = cnt % ncols
        ax = axs[row, col]

        if cell_annot is not None:
            for j in range(len(cell_annot)):
                filt = adata.obs[color] == cell_annot[j]
                filt = np.ravel(filt)
                ax.scatter(
                    X_org[filt, 0],
                    X_org[filt, 1],
                    c=colors[j],
                    s=pointsize,
                    alpha=alpha,
                )
        else:
            ax.scatter(
                X_org[:, 0],
                X_org[:, 1],
                c=colors,
                s=pointsize,
                alpha=alpha,
                cmap=cmap,
                edgecolor="none",
            )
        ax.quiver(
            X[:, 0],
            X[:, 1],
            V[:, 0],
            V[:, 1],
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=quiver_color,
            alpha=quiver_alpha,
        )

        ax.set_xlabel("Spliced")
        ax.set_ylabel("Unspliced")
        ax.set_title(f"{gene}")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax


__all__ = [
    "plot_velocity_phase",
    "gene_score_histogram",
    "velocity_embedding_stream",
    "scatter",
]
