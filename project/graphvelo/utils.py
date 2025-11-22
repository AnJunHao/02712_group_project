import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from anndata import AnnData
import scipy.sparse as sp
from scipy import sparse
from typing import List, Tuple

### This is Helper function to process a single row.
def get_row_knn(i: int, adj: sparse.csr_matrix, n_neighbors: int) -> Tuple[List[int], List[float]]: 
    # Determine target number of neighbors (excluding self)
    k_target = n_neighbors - 1
    
    # Fast slice using CSR internal structures
    # Accessing .indptr directly is much faster than adj[i, :]
    start, end = adj.indptr[i], adj.indptr[i+1]
    row_indices = adj.indices[start:end]
    row_data = adj.data[start:end]

    # Filter out self-loops
    mask = row_indices != i
    valid_idx = row_indices[mask]
    valid_wgt = row_data[mask]
    
    n_available = len(valid_idx)
    
    # optimized using argpartition
    if n_available > k_target:
        # Find smallest k elements 
        partitioned_idx = np.argpartition(valid_wgt, k_target)[:k_target]

        # Sort only these top k elements for consistent order
        top_k_local_order = np.argsort(valid_wgt[partitioned_idx])
        selection = partitioned_idx[top_k_local_order]
        
        final_idx = valid_idx[selection]
        final_wgt = valid_wgt[selection]
    else:
        sort_order = np.argsort(valid_wgt)
        final_idx = valid_idx[sort_order]
        final_wgt = valid_wgt[sort_order]

    # Construct result: [Self, Neighbor_1, ... Neighbor_k, 0, 0...]
    curr_idx = [i]
    curr_wgt = [0.0]
    
    # Add found neighbors
    curr_idx.extend(final_idx)
    curr_wgt.extend(final_wgt)
    
    # Pad with zeros if length is less than n_neighbors
    pad_len = n_neighbors - len(curr_idx)
    if pad_len > 0:
        curr_idx.extend([0] * pad_len)
        curr_wgt.extend([0.0] * pad_len)
        
    return curr_idx, curr_wgt

### Main function: Convert sparse adjacency matrix to KNN index and weight matrices.
def adj_to_knn(adj: sparse.spmatrix, n_neighbors: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    
    # Ensure matrix is in correct format
    if not sparse.isspmatrix_csr(adj):
        adj = adj.tocsr()

    n_obs = adj.shape[0]

    # Call external function using list comprehension
    results = [get_row_knn(i, adj, n_neighbors) for i in range(n_obs)]

    # Cobime the result
    list_idx, list_wgt = zip(*results)

    return np.array(list_idx, dtype=int), np.array(list_wgt, dtype=adj.dtype)


# Function for mack gene calculation
def calculate_mack_score_numba(x, v, nbrs_idx, t, eps=1e-5):
    score = np.zeros(x.shape[0])
    # for each cell
    for i in range(x.shape[0]):
        nbrs = nbrs_idx[i]
        # for each neighbor
        for j in range(len(nbrs)):
            # expression change
            d_x = x[i] - x[nbrs[j]]
            # Pseudotime change
            d_t = t[i] - t[nbrs[j]] + eps # avoid division by zero
            
            # calculate teh sign (should be 1, -1, or 0, I think d_x do not ahve nan)
            sign = d_x / d_t
            if sign > 0:
                sign = 1
            elif sign < 0:
                sign = -1
            elif sign == 0:
                sign = 0
            
            s_v = v[i]
            if s_v > 0:
                s_v = 1
            elif s_v < 0:
                s_v = -1
            elif s_v == 0:
                s_v = 0

            if sign == s_v:
                score[i] += 1
            else:
                score[i] += 0

        score[i] /= len(nbrs)
    return score


from pynndescent import NNDescent # for "pynn", "umap"
from sklearn.neighbors import NearestNeighbors # for "ball_tree", "kd_tree"
def knn(
    X: np.ndarray,
    k: int,
#    query_X: Optional[np.ndarray] = None,
    method: str = "kd-tree", # options: "kd-tree", "PyNNDescent" and "Ball tree"
#    exclude_self: bool = True,
#    knn_dim: int = 10,
#    pynn_num: int = 5000,
#    pynn_dim: int = 2,
    pynn_rand_state: int = 0,
#    n_jobs: int = -1,
#    return_nbrs: bool = False,
#    **kwargs,
):
    
    if method.lower() in ["pynn", "umap"]:
        nbrs = NNDescent(X, n_neighbors= k + 1, random_state= pynn_rand_state)
        nbrs_idx, distances = nbrs.query(X, k=k+1)
    elif method.lower() in ["ball_tree", "kd_tree"]:
        nbrs = NearestNeighbors(n_neighbors= k+1, algorithm= method.lower()).fit(X)
        nbrs_idx, distances = nbrs.kneighbors(X, n_neighbors=k + 1, return_distance=True)
    else:
        print("Using default method kd-tree for knn")
        nbrs = NearestNeighbors(n_neighbors= k+1, algorithm= "kd-tree").fit(X)
        nbrs_idx, distances = nbrs.kneighbors(X, n_neighbors=k + 1, return_distance=True)

    # remove self-neighbour
    nbrs_idx = nbrs_idx[:, 1:]
    distances = distances[:, 1:]

    return nbrs_idx, distances, nbrs, method

def mack_score(
    adata: AnnData,
    n_neighbors: int | None = None,
    basis: str | None = None,
    tkey: str | None = None,
    genes: list | None = None,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    X_data: NDArray[np.float64] | None = None,
    V_data: NDArray[np.float64] | None = None,
    n_jobs: int = -1,
    add_prefix: str | None = None,
    return_score: bool = False,
) -> pd.DataFrame | None:
    
         # Determine the number of jobs to use.
    if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
            n_jobs > os.cpu_count()):
        n_jobs = os.cpu_count()

    # Restrict genes if provided.
    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("No genes from your genes list appear in your adata object.")
    else:
        tmp_V = adata.layers[vkey].A if sp.issparse(adata.layers[vkey]) else adata.layers[vkey]
        genes = adata[:, ~np.isnan(tmp_V.sum(0))].var_names

    # Get X_data and V_data if not provided.
    if X_data is None or V_data is None:
        X_data = adata[:, genes].layers[ekey]
        V_data = adata[:, genes].layers[vkey]
    else:
        if V_data.shape[1] != X_data.shape[1] or len(genes) != X_data.shape[1]:
            raise ValueError(
                f"When providing X_data, a list of gene names that corresponds to the columns of X_data "
                f"must be provided")
    
    # Get kNN indices.
    if n_neighbors is None:
        nbrs_idx = adata.uns['neighbors']['indices']
    else:
        basis_for_knn = 'X_' + basis
        if basis_for_knn in adata.obsm.keys():
#            logging.info(f"Compute knn in {basis.upper()} basis...")
            nbrs_idx, _, _, _ = knn(adata.obsm[basis_for_knn], n_neighbors)
        else:
#            logging.info(f"Compute knn in original basis...")
            X_for_knn = adata.X.A if sp.issparse(adata.X) else adata.X
            nbrs_idx, _, _, _ = knn(X_for_knn, n_neighbors)


    # Get pseudotime
    t_annot = adata.obs[tkey]

    if hasattr(t_annot, "to_numpy"):
        t_data = t_annot.to_numpy()
    else:
        t_data = np.array(t_annot)

    t = t_data.flatten()
    # Compute MacK score per gene
    rows = []
    for g in genes:
        x = X_data[:, g].flatten()
        v = V_data[:, g].flatten()
        scores = calculate_mack_score_numba(x, v, nbrs_idx, t)
        rows.append((g, np.mean(scores)))


    mack_score_results = pd.DataFrame(rows, columns=["gene_name", "mack_score"])

        



    return gv_mack_score(
        adata,
        n_neighbors,
        basis,
        tkey,
        genes,
        ekey,
        vkey,
        X_data,
        V_data,
        n_jobs,
        add_prefix,
        return_score,
    )
