from scvelo.tools import recover_dynamics as scv_recover_dynamics
from scvelo.tools import velocity as scv_velocity
from scvelo.tools import latent_time as scv_latent_time
from anndata import AnnData
from typing import Any, Literal
from numpy.typing import NDArray

def recover_dynamics(
    data: AnnData,
    var_names: str | list[str] = "velocity_genes",
    n_top_genes: int | None = None,
    max_iter: int = 10,
    assignment_mode: str = "projection",
    t_max: float | bool | None = None,
    fit_time: bool | float | None = True,
    fit_scaling: bool | float | None = True,
    fit_steady_states: bool | None = True,
    fit_connected_states: bool | None = None,
    fit_basal_transcription: bool | None = None,
    use_raw: bool = False,
    load_pars: bool | None = None,
    return_model: bool | None = None,
    plot_results: bool = False,
    steady_state_prior: list[bool] | None = None,
    add_key: str = "fit",
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> AnnData | None:
    return scv_recover_dynamics(
        data=data,
        var_names=var_names,  # type: ignore
        n_top_genes=n_top_genes,
        max_iter=max_iter,
        assignment_mode=assignment_mode,
        t_max=t_max,
        fit_time=fit_time,  # type: ignore
        fit_scaling=fit_scaling,  # type: ignore
        fit_steady_states=fit_steady_states,  # type: ignore
        fit_connected_states=fit_connected_states,
        fit_basal_transcription=fit_basal_transcription,
        use_raw=use_raw,
        load_pars=load_pars,
        return_model=return_model,
        plot_results=plot_results,
        steady_state_prior=steady_state_prior,
        add_key=add_key,
        copy=copy,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
        **kwargs,
    )

def velocity(
    data: AnnData,
    vkey: str = "velocity",
    mode: Literal["deterministic", "stochastic", "dynamical"] = "stochastic",
    fit_offset: bool = False,
    fit_offset2: bool = False,
    filter_genes: bool = False,
    groups: str | list | None = None,
    groupby: str | list | NDArray | None = None,
    groups_for_fit: str | list | NDArray | None = None,
    constrain_ratio: float | tuple[float, ...] | None = None,
    use_raw: bool = False,
    use_latent_time: bool | None = None,
    perc: float | None = None,
    min_r2: float = 1e-2,
    min_likelihood: float = 1e-3,
    r2_adjusted: bool | None = None,
    use_highly_variable: bool = True,
    diff_kinetics: Any = None,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData | None:
    return scv_velocity(
        data=data,
        vkey=vkey,
        mode=mode,
        fit_offset=fit_offset,
        fit_offset2=fit_offset2,
        filter_genes=filter_genes,
        groups=groups,
        groupby=groupby,
        groups_for_fit=groups_for_fit,
        constrain_ratio=constrain_ratio,
        use_raw=use_raw,
        use_latent_time=use_latent_time,
        perc=perc,
        min_r2=min_r2,
        min_likelihood=min_likelihood,
        r2_adjusted=r2_adjusted,
        use_highly_variable=use_highly_variable,
        diff_kinetics=diff_kinetics,
        copy=copy,
        **kwargs,
    )

def latent_time(
    data: AnnData,
    vkey: str = "velocity",
    min_likelihood: float = 0.1,
    min_confidence: float = 0.75,
    min_corr_diffusion: float | None = None,
    weight_diffusion: float | None = None,
    root_key: str | None = None,
    end_key: str | None = None,
    t_max: float | None = None,
    copy: bool = False,
) -> AnnData | None:
    return scv_latent_time(
        data=data,
        vkey=vkey,
        min_likelihood=min_likelihood,
        min_confidence=min_confidence,
        min_corr_diffusion=min_corr_diffusion,
        weight_diffusion=weight_diffusion,
        root_key=root_key,
        end_key=end_key,
        t_max=t_max,
        copy=copy,
    )