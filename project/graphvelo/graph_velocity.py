import numpy as np
from anndata import AnnData
from numpy.typing import NDArray

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
        super().__init__(
            adata=adata,
            xkey=xkey,
            vkey=vkey,
            X_data=X_data,
            V_data=V_data,
            gene_subset=gene_subset,
            approx=approx,
            n_pcs=n_pcs,
            mo=mo,
        )

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