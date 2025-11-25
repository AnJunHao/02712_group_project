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

##Learn phi coefficients (transition weights) in tangent space.
def tangent_space_projection(
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 0.0,
    r: float = 1.0,
    loss_func: str = "linear",
    n_jobs: int = None,
):
    max_jobs = os.cpu_count() ## Determine number of jobs
    if not isinstance(n_jobs, int) or n_jobs <= 0 or n_jobs > max_jobs:
        n_jobs = max_jobs

    valid_genes = ~np.isnan(V.sum(axis=0)) # Filter genes with NaN velocity
    X = X[:, valid_genes]
    V = V[:, valid_genes]

    n_cells = X.shape[0]
    E = np.zeros((n_cells, n_cells), dtype=float)


    ## Parallel phi regression
    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(regression_phi)(
            i, X, V, C, nbrs, a, b, r, loss_func
        )
        for i in tqdm(
            range(n_cells),
            total=n_cells,
            desc="Learning Phi in tangent space projection.",
        )
    )

    # Fill phi matrix
    for i, phi_i in parallel_results:
        E[i, nbrs[i]] = phi_i

    return E

class GraphVelo():
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
    ####Train the GraphVelo model by learning the phi coefficients in tangent space.
    def train(
        self,
        a: float = 1,
        b: float = 10,
        r: float = 1,
        loss_func: str | None = None,
        transition_matrix: NDArray | None = None,
        softmax_adjusted: bool = False,
        n_jobs: int | None = None,
    ):
         # Decide loss function
        loss_func = loss_func or ("linear" if self.approx else "log")
        # Store parameters
        train_params = {
        "a": a,
        "b": b,
        "r": r,
        "loss_func": loss_func,
        "softmax_adjusted": softmax_adjusted,
        }
        # Compute transition matrix if not provided
        if transition_matrix is None:
            P = corr_kernel(
            self.X,
            self.V,
            self.nbrs_idx,
            corr_func=cos_corr,
            softmax_adjusted=softmax_adjusted,
            )
            P_dc = density_corrected_transition_matrix(P).A
        else:
            P_dc = transition_matrix

        # Tangent space projection
        T = tangent_space_projection(
        X=self.X,
        V=self.V,
        C=P_dc,
        nbrs_idx=self.nbrs_idx,
        a=a,
        b=b,
        r=r,
        loss_func=loss_func,
        n_jobs=n_jobs,
        )
        self.T = sp.csr_matrix(T)
        self.params.update(train_params)

    #Project the velocity vectors onto a low-dimensional embedding.
    def project_velocity(
        self, X_embedding: NDArray[np.float64], T: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        if T is None:
            T = self.T
        else:
            logging.warning(
            "You are projecting the velocity vectors with an external `phi` basis."
            )
        n_cells, n_dim = T.shape[0], X_embedding.shape[1]
        delta_X = np.zeros((n_cells, n_dim))
    
        sparse_input = sp.issparse(X_embedding)
        if sparse_input:
            X_embedding = X_embedding.A
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in tqdm(
                range(n_cells),
                total=n_cells,
                desc="projecting velocity vectors to embedding",
            ):
                idx = T[i].indices            
                diff = X_embedding[idx] - X_embedding[i]  
                diff = np.nan_to_num(diff)     

                weights = T[i].data            
                delta_X[i] = weights.dot(diff) 

        return sp.csr_matrix(delta_X) if sparse_input else delta_X

    ####Plot the distribution of the learned phi coefficients.
    def plot_phi_dist(self) -> None:
        T = self.T.A
        sns.distplot(T[T>0])
        plt.show()
    # Write the learned phi coefficients to the AnnData object.
    def write_to_adata(self, adata: AnnData, key: str | None = None) -> None:

        key = key or "gv"   
        params_key = f"{key}_params"
        existing = adata.uns.get(params_key, {})
        adata.uns[params_key] = {**existing, **self.params}
        adata.obsp[key] = self.T.copy()


