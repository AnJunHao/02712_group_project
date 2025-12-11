import numpy as np
import tqdm
import scipy.sparse as sp

def cross_boundary_correctness_one_transition(adata, transition, V_data,annotation_key = "celltype", reduce_space_key = "umap", neighbor_key = "neighbors"):
    set_A = np.where(adata.obs[annotation_key] == transition[0])[0]
    set_B = np.where(adata.obs[annotation_key] == transition[1])[0]
    
    nbrs_idx = adata.uns[neighbor_key]['indices']
#    nbrs = nbrs_idx[set_A]
#    inter_set = np.intersect1d(np.array(set_B), np.array(nbrs.flatten()))
    CBC = 0.0
    for i in set_A:
        nbrs_i = nbrs_idx[i]
        intersect = np.intersect1d(np.array(set_B), np.array(nbrs_i))
        if len(intersect) == 0:
            continue   

        space = None
        if reduce_space_key is None:
            space = adata.X.A if sp.issparse(adata.X) else adata.X

        else:
            space = adata.obsm[reduce_space_key]


        sum_cos_sim = 0.0
        for j in intersect:
            x_ji = space[j] - space[i]
            v_i = V_data[i]
            cos_sim = cosine_similarity(x_ji, v_i)
            sum_cos_sim += cos_sim
        avg_cos_sim = sum_cos_sim / len(intersect)
        CBC += avg_cos_sim

    CBC = CBC / len(set_A)    
    return CBC


def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot_product / (norm_u * norm_v)


def calculate_degradation_graphvelo(adata, splice_key = "Ms", unsplice_key = "Mu", velocity_key = "velocity_gv", degradation_key = "graphvelo_degradation"):
    adata.layers[degradation_key]  = ((adata.layers[unsplice_key] - adata.layers[velocity_key]) / adata.layers[splice_key])
