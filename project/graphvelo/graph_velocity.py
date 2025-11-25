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
    Solve phi_i (weights to neighbors) for cell i by minimizing 
    L(w) = a||Dw - v||^2 - b <w, c> + r||w||^2
    """

    # Extract local data
    x = X[i]
    v = V[i]
    idx = nbrs[i]

    # Correlation weights
    c = C[i, idx]
    c_norm = np.linalg.norm(c)
    if c_norm > 1e-12:
        c = c / c_norm

    # Local direction vectors
    D = X[idx] - x

    # Normalize distances if needed
    if norm_dist:
        dist = np.linalg.norm(D, axis=1)
        dist[dist == 0] = 1
        D = D / dist[:, None]

    # Build normal equation system:
    # (a DᵀD + rI) w = (a Dᵀv + b c)
    A = a * (D.T @ D) + r * np.eye(len(idx))
    b_vec = a * (D.T @ v) + b * c

    # Solve system
    try:
        w = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b_vec, rcond=None)[0]

    return i, w


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
    """
    GraphVelo encapsulates the workflow for learning a manifold-constrained velocity projection.
    
    It supports computing a low-dimensional representation of the gene expression space,
    estimating a velocity graph (phi coefficients), and transform velocity vectors to different basis.
    """

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
        """
        Initialize the GraphVelo object.
        
        Parameters:
            adata: AnnData object containing the expression and velocity data.
            xkey (str): Key in adata.layers for the expression data.
            vkey (str): Key in adata.layers for the velocity data.
            X_data, V_data: Optionally provided expression and velocity matrices.
            gene_subset: Optionally, a subset of genes to use.
            approx (bool): If True, perform an approximate projection using PCA.
            n_pcs (int): Number of principal components for dimensionality reduction.
            mo (bool): Flag indicating if multi-omic data is used (affects neighbor extraction).
        """

        if X_data is not None and V_data is not None:
            X = np.array(X_data.A if sp.issparse(X_data)
            else X_data)
            V = np.array(V_data.A if sp.issparse(V_data)
            else V_data)
        else:
            X_org = np.array(
                adata.layers[xkey].A
                if sp.issparse(adata.layers[xkey])
                else adata.layers[xkey]
            )
            V_org = np.array(
                adata.layers[vkey].A
                if sp.issparse(adata.layers[vkey])
                else adata.layers[vkey]
            )
            
            subset = np.ones(adata.n_vars, bool)
            if gene_subset is not None:
                var_names_subset = adata.var_names.isin(gene_subset)
                subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
            else: 
                gene_subset = adata.var_names
            X = X_org[:, subset]
            V = V_org[:, subset]

            nans = np.isnan(np.sum(V, axis=0))
            logging.info(f"{nans.sum()} genes are removed because of nan velocity values.")
            if np.any(nans):
                X = X[:, ~nans]
                V = V[:, ~nans]
        self.approx = False

        if "neighbors" not in adata.uns.keys():
            # Check construction knn in reduced space.
            if mo:
                if 'WNN' not in adata.uns.keys():
                    logging.error("`WNN` not in adata.uns")
                nbrs_idx = adata.uns['WNN']['indices']
            else:
                raise ValueError("Please run dyn.tl.neighbors first.")
        else:
            nbrs_idx = adata.uns["neighbors"]['indices']
        self.nbrs_idx = nbrs_idx
        
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
        else: 
            self.X = X
            self.V = V
        self.params = {'xkey': xkey,
                       'vkey': vkey,
                       'gene_subset': list(gene_subset),
                       'approx': approx,
                       'n_pcs': n_pcs}

    def train(self, a=1, b=10, r=1, loss_func=None, transition_matrix=None, softmax_adjusted=False, n_jobs=None):
        """
        Train the GraphVelo model by learning the phi coefficients in tangent space.
        
        Parameters:
            a, b, r: Weights for the loss function components.
            loss_func (str): Loss function type; defaults to 'linear' if approx is True, else 'log'.
            transition_matrix: Optionally provided transition matrix; if not provided, it is computed.
            softmax_adjusted (bool): Flag for adjusting the softmax in the correlation kernel.
        """

        if loss_func is None:
            loss_func = 'linear' if self.approx else 'log'
        train_params = {'a': a, 'b': b, 'r': r, 'loss_func': loss_func, 'softmax_adjusted': softmax_adjusted}
        if transition_matrix is None:
            P = corr_kernel(self.X, self.V, self.nbrs_idx, corr_func=cos_corr, softmax_adjusted=softmax_adjusted)
            P_dc = density_corrected_transition_matrix(P).A
        else:
            P_dc = transition_matrix
        T = tangent_space_projection(self.X, self.V, P_dc, self.nbrs_idx, a=a, b=b, r=r, loss_func=loss_func, n_jobs=n_jobs)
        self.T = sp.csr_matrix(T)
        self.params.update(train_params)
    
    def project_velocity(self, X_embedding, T=None) -> np.ndarray:
        """
        Project the velocity vectors onto a low-dimensional embedding.
        
        Parameters:
            X_embedding (np.ndarray): The low-dimensional embedding coordinates.
            T (optional): An external phi basis. If None, uses the learned phi coefficients.
        
        Returns:
            np.ndarray: The projected velocity vectors in the embedding space.
        """

        if T is None:
            T = self.T
        else:
            logging.warning('You are projecting the velocity vectors with an external `phi` basis.')
        n = T.shape[0]
        delta_X = np.zeros((n, X_embedding.shape[1]))

        sparse_emb = False
        if sp.issparse(X_embedding):
            X_embedding = X_embedding.A
            sparse_emb = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in tqdm(
                range(n),
                total=n,
                desc="projecting velocity vector to low dimensional embedding",
            ):
                idx = T[i].indices
                diff_emb = X_embedding[idx] - X_embedding[i, None]
                if np.isnan(diff_emb).sum() != 0:
                    diff_emb[np.isnan(diff_emb)] = 0
                T_i = T[i].data
                delta_X[i] = T_i.dot(diff_emb)

        return sp.csr_matrix(delta_X) if sparse_emb else delta_X
          def plot_phi_dist(self):
        """
        Plot the distribution of the learned phi coefficients.
        
        This uses seaborn for visualization.
        """
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "You need to install `seaborn` for `phi` distribution visualization.")

        T = self.T.A
        sns.distplot(T[T>0])
        plt.show()

    def write_to_adata(self, adata, key=None):
        """
        Write the learned phi coefficients (velocity projection basis) to the AnnData object.
        
        Parameters:
            adata: AnnData object where the phi matrix should be stored.
            key: The key under which the phi parameters will be saved in adata.
                 Defaults to 'gv' if not provided.
        """
        key = 'gv' if key is None else key
        adata.uns[f"{key}_params"] = {
            **adata.uns.get(f"{key}_params", {}),
            **self.params,
        }
        adata.obsp[key] = self.T.copy()
