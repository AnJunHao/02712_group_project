import os
import logging
import warnings
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
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from .tangent_space import corr_kernel, cos_corr, density_corrected_transition_matrix, _estimate_dt


def regression_phi(
    i: int, 
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 0.0,
    r: float = 1.0,
    loss_func: str = "linear",
    norm_dist: bool = False,
):
    """
    Compute the regression coefficients (phi) for cell i in the tangent space.
    
    Parameters:
        i (int): The index of the cell to process.
        X (np.ndarray): The coordinate matrix (cells x genes).
        V (np.ndarray): The velocity matrix (cells x genes).
        C (np.ndarray): The correlation (or transition) matrix (cells x ?).
        nbrs (list): List of neighbor indices for each cell.
        a (float): Weight for the reconstruction error term.
        b (float): Weight for the cosine similarity term.
        r (float): Weight for the regularization term.
        loss_func (str): Loss function type ('linear' or 'log').
        norm_dist (bool): If True, normalize the difference vectors.
    
    Returns:
        tuple: The cell index and the optimized weight vector (phi) for cell i.
    """
    x, v, c, idx = X[i], V[i], C[i], nbrs[i]
    c = c[idx]

    # normalized differences
    D = X[idx] - x
    if norm_dist:
        dist = np.linalg.norm(D, axis=1)
        dist[dist == 0] = 1
        D /= dist[:, None]

    # co-optimization
    c_norm = np.linalg.norm(c)

    def func(w):
        v_ = w @ D

        # cosine similarity between w and c
        if b == 0:
            sim = 0
        else:
            cw = c_norm * np.linalg.norm(w)
            if cw > 0:
                sim = c.dot(w) / cw
            else:
                sim = 0

        # reconstruction error between v_ and v
        rec = v_ - v
        rec = rec.dot(rec)
        if loss_func is None or loss_func == "linear":
            rec = rec
        elif loss_func == "log":
            rec = np.log(rec)
        else:
            raise NotImplementedError(
                f"The function {loss_func} is not supported. Choose either `linear` or `log`."
            )

        # regularization
        reg = 0 if r == 0 else w.dot(w)

        ret = a * rec - b * sim + r * reg
        return ret

    def fjac(w):
        v_ = w @ D

        # reconstruction error
        jac_con = 2 * a * D @ (v_ - v)

        if loss_func is None or loss_func == "linear":
            jac_con = jac_con
        elif loss_func == "log":
            jac_con = jac_con / (v_ - v).dot(v_ - v)

        # cosine similarity
        w_norm = np.linalg.norm(w)
        if w_norm == 0 or b == 0:
            jac_sim = 0
        else:
            jac_sim = b * (c / (w_norm * c_norm) - w.dot(c) / (w_norm**3 * c_norm) * w)

        # regularization
        if r == 0:
            jac_reg = 0
        else:
            jac_reg = 2 * r * w

        return jac_con - jac_sim + jac_reg

    res = minimize(func, x0=C[i, idx], jac=fjac)
    return i, res["x"]

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
    """
    The function generates a graph based on the velocity data by minimizing the loss function:
                    L(w_i) = a |v_ - v|^2 - b cos(u, v_) + lambda * \sum_j |w_ij|^2
    where v_ = \sum_j w_ij*d_ij. The flow from i- to j-th node is returned as the basis matrix phi[i, j],

    Arguments
    ---------
        X: :class:`~numpy.ndarray`
            The coordinates of cells in the expression space.
        V: :class:`~numpy.ndarray`
            The velocity vectors in the expression space.
        C: :class:`~numpy.ndarray`
            The transition matrix of cells based on the correlation/cosine kernel.
        nbrs: list
            List of neighbor indices for each cell.
        a: float (default 1.0)
            The weight for preserving the velocity length.
        b: float (default 1.0)
            The weight for the cosine similarity.
        r: float (default 1.0)
            The weight for the regularization.
        n_jobs: `int` (default: available threads)
            Number of parallel jobs.

    Returns
    -------
        E: :class:`~numpy.ndarray`
            The phi matrix.
    """
    if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
            n_jobs > os.cpu_count()):
        n_jobs = os.cpu_count()
    if isinstance(n_jobs, int):
        logging.info(f'running {n_jobs} jobs in parallel')

    vgenes = np.ones(V.shape[-1], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]
    X = X[:, vgenes]

    E = np.zeros((X.shape[0], X.shape[0]))

    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(regression_phi)(
            i, 
            X,
            V,
            C,
            nbrs,
            a,
            b,
            r,
            loss_func,
        )
        for i in tqdm(
        range(X.shape[0]),
        total=X.shape[0],
        desc="Learning Phi in tangent space projection.",
    ))

    for i, res_x in res:
        E[i][nbrs[i]] = res_x
    
    return E


class GraphVelo():
     def __init__(
        self,
        adata, 
        xkey='Ms', 
        vkey='velocity', 
        X_data=None,
        V_data=None,
        gene_subset=None,
        approx=True,
        n_pcs=30,
        mo=False,):
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
    def train(self, a=1, b=10, r=1, loss_func=None, transition_matrix=None, softmax_adjusted=False, n_jobs=None):
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
        nbrs=self.nbrs_idx,
        a=a,
        b=b,
        r=r,
        loss_func=loss_func,
        n_jobs=n_jobs,
        )
        self.T = sp.csr_matrix(T)
        self.params.update(train_params)

    #Project the velocity vectors onto a low-dimensional embedding.
    def project_velocity(self, X_embedding, T=None) -> np.ndarray:
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
    def plot_phi_dist(self):
        T = self.T.A
        sns.distplot(T[T>0])
        plt.show()

    # Write the learned phi coefficients to the AnnData object.
    def write_to_adata(self, adata, key = None):

        key = key or "gv"   
        params_key = f"{key}_params"
        existing = adata.uns.get(params_key, {})
        adata.uns[params_key] = {**existing, **self.params}
        adata.obsp[key] = self.T.copy()
