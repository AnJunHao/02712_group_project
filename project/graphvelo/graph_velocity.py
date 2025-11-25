import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm
from scipy.optimize import minimize
from joblib import Parallel, delayed
from .tangent_space import corr_kernel, cos_corr, density_corrected_transition_matrix, _estimate_dt
import os
import logging
import warnings



from graphvelo.graph_velocity import GraphVelo as GraphVeloBase


class GraphVelo(GraphVeloBase):
        def __init__(
        self,
        adata: AnnData,
        xkey: str = "Ms",
        vkey: str = "velocity",
        X_data: NDArray[np.float64] | None = None,
        V_data: NDArray[np.float64] | None = None,
        gene_subset: NDArray | list | None = None,
        approx: bool = True,
        n_pcs: int = 30,
        mo: bool = False,
    ) -> None:
         # Convert sparse matrix to dense.
        dense = lambda a: a.A if sp.issparse(a) else np.asarray(a)

        # Load data from provided arguments
        if X_data is not None and V_data is not None:
            X = dense(X_data)
            V = dense(V_data)
        else:
            X_org = dense(adata.layers[xkey])
            V_org = dense(adata.layers[vkey])
            X = X_org
            V = V_org

        # If no subset provided → use all genes
        if gene_subset is None:
            gene_subset = adata.var_names

        # Boolean mask over all genes
        subset = adata.var_names.isin(gene_subset)

        # Select genes
        X = X[:, subset]
        V = V[:, subset]

        # Identify genes with NaN velocity values
        nan_genes = np.isnan(V).any(axis=0)
        logging.info(f"{nan_genes.sum()} genes are removed because of NaN velocity values.")

        # Remove genes containing NaN
        if nan_genes.any():
            X = X[:, ~nan_genes]
            V = V[:, ~nan_genes]

        # Disable approx for now
        self.approx = False  

        # Retrieve neighbor graph indices
        if "neighbors" in adata.uns:
            nbrs_idx = adata.uns["neighbors"]["indices"]

        elif mo:
            # Multi-omics mode: use Weighted Nearest Neighbors (WNN)
            if "WNN" not in adata.uns:
                logging.error("`WNN` not in adata.uns")
            nbrs_idx = adata.uns["WNN"]["indices"]

        else:
            raise ValueError("Please run dyn.tl.neighbors first.")

        self.nbrs_idx = nbrs_idx

        # Approximation mode: project velocity into PCA space
        if approx:
            self.approx = True
            dt = _estimate_dt(X, V, nbrs_idx)

            X_plus_V = X + V * dt
            X_plus_V[X_plus_V < 0] = 0

            X = np.log1p(X)
            X_plus_V = np.log1p(X_plus_V)

            pca = PCA(n_components=n_pcs, svd_solver='arpack', random_state=0)
            pca_fit = pca.fit(X)

            X_pca = pca_fit.transform(X)
            Y_pca = pca_fit.transform(X_plus_V)
            V_pca = (Y_pca - X_pca) / dt

            self.X = X_pca
            self.V = V_pca
        # Non-approximation mode: use original data
        else:
            self.X = X
            self.V = V

        # Store parameters
        self.params = {
            "xkey": xkey,
            "vkey": vkey,
            "gene_subset": list(gene_subset),
            "approx": approx,
            "n_pcs": n_pcs,
        }
    def train(
        self,
        a: int = 1,
        b: int = 10,
        r: int = 1,
        loss_func: str | None = None,
        transition_matrix: NDArray[np.float64] | None = None,
        softmax_adjusted: bool = False,
    ) -> None:
        super().train(
            a=a,
            b=b,
            r=r,
            loss_func=loss_func,
            transition_matrix=transition_matrix,
            softmax_adjusted=softmax_adjusted,
        )

    def project_velocity(
        self, X_embedding: NDArray[np.float64], T: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        return super().project_velocity(X_embedding, T=T)

    def plot_phi_dist(self) -> None:
        super().plot_phi_dist()

    def write_to_adata(self, adata: AnnData, key: str | None = None) -> None:
        super().write_to_adata(adata, key=key)