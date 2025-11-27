# type: ignore
"""EM Model Utilities for Dynamical Model Recovery

This module contains helper functions and the BaseDynamics class for
recovering RNA velocity dynamics using an expectation-maximization approach.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats.distributions import chi2, norm

import matplotlib as mpl
import matplotlib.pyplot as pl
from matplotlib import gridspec, rcParams

from scvelo import logging as logg
from scvelo.core import clipped_log, invert, SplicingDynamics
from scvelo.preprocessing.moments import get_connectivities

from .utils import make_dense, round

exp = np.exp


"""Helper functions"""


def normalize(X, axis=0, min_confidence=None):
    """Normalize array along specified axis."""
    X_sum = np.sum(X, axis=axis)
    if min_confidence:
        X_sum += min_confidence
    X_sum += X_sum == 0
    return X / X_sum


def convolve(x, weights=None):
    """Convolve input with weights."""
    if weights is None:
        return x
    else:
        return weights.multiply(x).tocsr() if issparse(weights) else weights * x


def linreg(u, s):
    """Linear regression fit."""
    ss_ = s.multiply(s).sum(0) if issparse(s) else (s**2).sum(0)
    us_ = s.multiply(u).sum(0) if issparse(s) else (s * u).sum(0)
    return us_ / ss_


def compute_dt(t, clipped=True, axis=0):
    """Compute time differences."""
    prepend = np.min(t, axis=axis)[None, :]
    dt = np.diff(np.sort(t, axis=axis), prepend=prepend, axis=axis)
    m_dt = np.max([np.mean(dt, axis=axis), np.max(t, axis=axis) / len(t)], axis=axis)
    m_dt = np.clip(m_dt, 0, None)
    if clipped:  # Poisson upper bound
        ub = m_dt + 3 * np.sqrt(m_dt)
        dt = np.clip(dt, 0, ub)
    return dt


def root_time(t, root=None):
    """Root time alignment."""
    nans = np.isnan(np.sum(t, axis=0))
    if np.any(nans):
        t = t[:, ~nans]

    t_root = 0 if root is None else t[root]
    o = np.array(t >= t_root, dtype=int)
    t_after = (t - t_root) * o
    t_origin = np.max(t_after, axis=0)
    t_before = (t + t_origin) * (1 - o)

    t_switch = np.min(t_before, axis=0)
    t_rooted = t_after + t_before
    return t_rooted, t_switch


def compute_shared_time(t, perc=None, norm=True):
    """Compute shared latent time across genes."""
    nans = np.isnan(np.sum(t, axis=0))
    if np.any(nans):
        t = np.array(t[:, ~nans])
    t -= np.min(t)

    tx_list = np.percentile(t, [15, 20, 25, 30, 35] if perc is None else perc, axis=1)
    tx_max = np.max(tx_list, axis=1)
    tx_max += tx_max == 0
    tx_list /= tx_max[:, None]

    mse = []
    for tx in tx_list:
        tx_ = np.sort(tx)
        linx = np.linspace(0, 1, num=len(tx_))
        mse.append(np.sum((tx_ - linx) ** 2))
    idx_best = np.argsort(mse)[:2]

    t_shared = tx_list[idx_best].sum(0)
    if norm:
        t_shared /= t_shared.max()

    return t_shared


"""Dynamics delineation"""


def unspliced(tau, u0, alpha, beta):
    """Unspliced mRNA at time tau."""
    expu = exp(-beta * tau)
    return u0 * expu + alpha / beta * (1 - expu)


def spliced(tau, s0, u0, alpha, beta, gamma):
    """Spliced mRNA at time tau."""
    c = (alpha - u0 * beta) * invert(gamma - beta)
    expu, exps = exp(-beta * tau), exp(-gamma * tau)
    return s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu)


def adjust_increments(tau, tau_=None):
    """Adjust time increments to avoid meaningless jumps."""
    tau_new = np.array(tau)
    tau_ord = np.sort(tau_new)
    dtau = np.diff(tau_ord, prepend=0)

    if tau_ is None:
        ub = 3 * np.percentile(dtau, 99.5, axis=0)
    else:
        tau_new_ = np.array(tau_)
        tau_ord_ = np.sort(tau_new_)
        dtau_ = np.diff(tau_ord_, prepend=0)
        ub = 3 * np.percentile(np.hstack([dtau, dtau_]), 99.5, axis=0)

        idx = np.where(dtau_ > ub)[0]
        for i in idx:
            ti, dti = tau_ord_[i], dtau_[i]
            tau_new_[tau_ >= ti] -= dti

    idx = np.where(dtau > ub)[0]
    for i in idx:
        ti, dti = tau_ord[i], dtau[i]
        tau_new[tau >= ti] -= dti

    return tau_new if tau_ is None else (tau_new, tau_new_)


def tau_inv(u, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None):
    """Inverse function to recover time from state."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv_u = (gamma >= beta) if gamma is not None else True
        inv_us = np.invert(inv_u)
    any_invu = np.any(inv_u) or s is None
    any_invus = np.any(inv_us) and s is not None

    if any_invus:  # tau_inv(u, s)
        beta_ = beta * invert(gamma - beta)
        xinf = alpha / gamma - beta_ * (alpha / beta)
        tau = (
            -1 / gamma * clipped_log((s - beta_ * u - xinf) / (s0 - beta_ * u0 - xinf))
        )

    if any_invu:  # tau_inv(u)
        uinf = alpha / beta
        tau_u = -1 / beta * clipped_log((u - uinf) / (u0 - uinf))
        tau = tau_u * inv_u + tau * inv_us if any_invus else tau_u
    return tau


def assign_tau(
    u, s, alpha, beta, gamma, t_=None, u0_=None, s0_=None, assignment_mode=None
):
    """Assign time points to observations."""
    if assignment_mode in {"full_projection", "partial_projection"} or (
        assignment_mode == "projection" and beta < gamma
    ):
        x_obs = np.vstack([u, s]).T
        t0 = tau_inv(np.min(u[s > 0]), u0=u0_, alpha=0, beta=beta)

        num = np.clip(int(len(u) / 5), 200, 500)
        tpoints = np.linspace(0, t_, num=num)
        tpoints_ = np.linspace(0, t0, num=num)[1:]
        xt = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(tpoints)
        xt_ = SplicingDynamics(
            alpha=0, beta=beta, gamma=gamma, initial_state=[u0_, s0_]
        ).get_solution(tpoints_)

        # assign time points (oth. projection onto 'on' and 'off' curve)
        tau = tpoints[
            ((xt[None, :, :] - x_obs[:, None, :]) ** 2).sum(axis=2).argmin(axis=1)
        ]
        tau_ = tpoints_[
            ((xt_[None, :, :] - x_obs[:, None, :]) ** 2).sum(axis=2).argmin(axis=1)
        ]
    else:
        tau = tau_inv(u, s, 0, 0, alpha, beta, gamma)
        tau = np.clip(tau, 0, t_)

        tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma)
        tau_ = np.clip(tau_, 0, np.max(tau_[s > 0]))

    return tau, tau_, t_


def compute_divergence(
    u,
    s,
    alpha,
    beta,
    gamma,
    scaling=1,
    t_=None,
    u0_=None,
    s0_=None,
    tau=None,
    tau_=None,
    std_u=1,
    std_s=1,
    normalized=False,
    mode="distance",
    assignment_mode=None,
    var_scale=False,
    kernel_width=None,
    fit_steady_states=True,
    connectivities=None,
    constraint_time_increments=True,
    reg_time=None,
    reg_par=None,
    min_confidence=None,
    pval_steady=None,
    steady_u=None,
    steady_s=None,
    noise_model="chi",
    time_connectivities=None,
    clusters=None,
    **kwargs,
):
    """Estimates the divergence of ODE to observations.

    Available metrics: distance, mse, likelihood, loglikelihood.
    """
    # set tau, tau_
    if u0_ is None or s0_ is None:
        u0_, s0_ = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
            t_, stacked=False
        )
    if tau is None or tau_ is None or t_ is None:
        tau, tau_, t_ = assign_tau(
            u, s, alpha, beta, gamma, t_, u0_, s0_, assignment_mode
        )

    std_u /= scaling

    # adjust increments of tau, tau_ to avoid meaningless jumps
    if constraint_time_increments:
        ut, st = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
            tau, stacked=False
        )
        ut_, st_ = SplicingDynamics(
            alpha=0, beta=beta, gamma=gamma, initial_state=[u0_, s0_]
        ).get_solution(tau_, stacked=False)

        distu, distu_ = (u - ut) / std_u, (u - ut_) / std_u
        dists, dists_ = (s - st) / std_s, (s - st_) / std_s

        res = np.array([distu_**2 + dists_**2, distu**2 + dists**2])
        if connectivities is not None and connectivities is not False:
            res = (
                np.array([connectivities.dot(r) for r in res])
                if res.ndim > 2
                else connectivities.dot(res.T).T
            )

        o = np.argmin(res, axis=0)

        off, on = o == 0, o == 1
        if np.any(on) and np.any(off):
            tau[on], tau_[off] = adjust_increments(tau[on], tau_[off])
        elif np.any(on):
            tau[on] = adjust_increments(tau[on])
        elif np.any(off):
            tau_[off] = adjust_increments(tau_[off])

    # compute induction/repression state distances
    ut, st = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
        tau, stacked=False
    )
    ut_, st_ = SplicingDynamics(
        alpha=0, beta=beta, gamma=gamma, initial_state=[u0_, s0_]
    ).get_solution(tau_, stacked=False)

    if ut.ndim > 1 and ut.shape[1] == 1:
        ut = np.ravel(ut)
        st = np.ravel(st)
    if ut_.ndim > 1 and ut_.shape[1] == 1:
        ut_ = np.ravel(ut_)
        st_ = np.ravel(st_)

    distu, distu_ = (u - ut) / std_u, (u - ut_) / std_u
    dists, dists_ = (s - st) / std_s, (s - st_) / std_s

    if mode == "unspliced_dists":
        return distu, distu_

    elif mode == "outside_of_trajectory":
        return np.sign(distu) * np.sign(distu_) == 1

    distx = distu**2 + dists**2
    distx_ = distu_**2 + dists_**2

    res, varx = np.array([distx_, distx]), 1  # default vals

    if noise_model == "normal":
        if var_scale:
            o = np.argmin([distx_, distx], axis=0)
            varu = np.nanvar(distu * o + distu_ + (1 - o), axis=0)
            vars = np.nanvar(dists * o + dists_ + (1 - o), axis=0)

            distx = distu**2 / varu + dists**2 / vars
            distx_ = distu_**2 / varu + dists_**2 / vars

            varx = varu * vars

            std_u *= np.sqrt(varu)
            std_s *= np.sqrt(vars)

    # compute steady state distances
    if fit_steady_states:
        u_inf, s_inf = alpha / beta, alpha / gamma
        distx_steady = ((u - u_inf) / std_u) ** 2 + ((s - s_inf) / std_s) ** 2
        distx_steady_ = (u / std_u) ** 2 + (s / std_s) ** 2
        res = np.array([distx_, distx, distx_steady_, distx_steady])

    if connectivities is not None and connectivities is not False:
        res = (
            np.array([connectivities.dot(r) for r in res])
            if res.ndim > 2
            else connectivities.dot(res.T).T
        )

    # compute variances
    if noise_model == "chi":
        if var_scale:
            o = np.argmin([distx_, distx], axis=0)
            dist = distx * o + distx_ * (1 - o)
            sign = np.sign(dists * o + dists_ * (1 - o))
            varx = np.mean(dist, axis=0) - np.mean(sign * np.sqrt(dist), axis=0) ** 2
            if kernel_width is not None:
                varx *= kernel_width**2
            res /= varx
        elif kernel_width is not None:
            res /= kernel_width**2

    if reg_time is not None and len(reg_time) == len(distu_):
        o = np.argmin(res, axis=0)
        t_max = (t_ + tau_) * (o == 0)
        t_max /= np.max(t_max, axis=0)
        reg_time /= np.max(reg_time)

        dist_tau = (tau - reg_time[:, None]) ** 2
        dist_tau_ = (tau_ + t_ - reg_time[:, None]) ** 2
        mu_res = np.mean(res, axis=1)
        if reg_par is not None:
            mu_res *= reg_par

        res[0] += dist_tau_ * mu_res[0]
        res[1] += dist_tau * mu_res[1]
        if fit_steady_states:
            res[2] += dist_tau * mu_res[1]
            res[3] += dist_tau_ * mu_res[0]

    if mode == "tau":
        res = [tau, tau_]

    elif mode == "likelihood":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)

    elif mode == "nll":
        res = np.log(2 * np.pi * np.sqrt(varx)) + 0.5 * res
        if normalized:
            res = normalize(res, min_confidence=min_confidence)

    elif mode == "confidence":
        res = np.array([res[0], res[1]])
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = np.median(
            np.max(res, axis=0) - (np.sum(res, axis=0) - np.max(res, axis=0)), axis=1
        )

    elif mode in {"soft_eval", "soft"}:
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)

        o_, o = res[0], res[1]
        res = np.array([o_, o, ut * o + ut_ * o_, st * o + st_ * o_])

    elif mode in {"hardsoft_eval", "hardsoft"}:
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        o = np.argmax(res, axis=0)
        o_, o = (o == 0) * res[0], (o == 1) * res[1]
        res = np.array([o_, o, ut * o + ut_ * o_, st * o + st_ * o_])

    elif mode in {"hard_eval", "hard"}:
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        o = np.argmax(res, axis=0)
        o_, o = o == 0, o == 1
        res = np.array([o_, o, ut * o + ut_ * o_, st * o + st_ * o_])

    elif mode == "soft_state":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = res[1] - res[0]

    elif mode == "hard_state":
        res = np.argmin(res, axis=0)

    elif mode == "steady_state":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = res[2] + res[3]

    elif mode in {"assign_timepoints", "time"}:
        o = np.argmin(res, axis=0)

        tau_ *= o == 0
        tau *= o == 1

        if 2 in o:
            o[o == 2] = 1
        if 3 in o:
            o[o == 3] = 0

        t = tau * (o == 1) + (tau_ + t_) * (o == 0)
        res = [t, tau, o] if mode == "assign_timepoints" else t

    elif mode == "dists":
        o = np.argmin(res, axis=0)

        tau_ *= o == 0
        tau *= o == 1

        if 2 in o:
            o[o == 2] = 1
        if 3 in o:
            o[o == 3] = 0

        distu = distu * (o == 1) + distu_ * (o == 0)
        dists = dists * (o == 1) + dists_ * (o == 0)
        res = distu, dists

    elif mode == "distx":
        o = np.argmin(res, axis=0)

        tau_ *= o == 0
        tau *= o == 1

        if 2 in o:
            o[o == 2] = 1
        if 3 in o:
            o[o == 3] = 0

        distu = distu * (o == 1) + distu_ * (o == 0)
        dists = dists * (o == 1) + dists_ * (o == 0)
        res = distu**2 + dists**2

    elif mode == "gene_likelihood":
        o = np.argmin(res, axis=0)

        tau_ *= o == 0
        tau *= o == 1

        if 2 in o:
            o[o == 2] = 1
        if 3 in o:
            o[o == 3] = 0

        distu = distu * (o == 1) + distu_ * (o == 0)
        dists = dists * (o == 1) + dists_ * (o == 0)

        idx = np.array((u > np.max(u, 0) / 5) & (s > np.max(s, 0) / 5), dtype=int)
        idx = idx / idx
        distu *= idx
        dists *= idx

        distx = distu**2 + dists**2

        # compute variance / equivalent to np.var(np.sign(sdiff) * np.sqrt(distx))
        varx = (
            np.nanmean(distx, 0) - np.nanmean(np.sign(dists) * np.sqrt(distx), 0) ** 2
        )

        if clusters is not None:
            res = []
            for cat in clusters.cat.categories:
                idx_cat = np.array(clusters == cat)
                distx_cat = distu[idx_cat] ** 2 + dists[idx_cat] ** 2
                distx_sum = np.nansum(distx_cat, 0)

                # penalize if very low count number
                n = np.sum(np.invert(np.isnan(distx_cat)), 0) - len(distx_cat) * 0.01
                n = np.clip(n, 2, None)
                distx_sum[n < np.nanmax(n) / 5] = np.nanmax(distx_sum)
                ll = -1 / 2 / n * distx_sum / varx - 1 / 2 * np.log(2 * np.pi * varx)
                ll[distx_sum == 0] = np.nan
                res.append(ll)
            res = np.exp(res)
        else:
            n = np.clip(len(distu) - len(distu) * 0.01, 2, None)
            ll = -1 / 2 / n * np.nansum(distx, 0) / varx
            ll -= 1 / 2 * np.log(2 * np.pi * varx)
            res = np.exp(ll)

    elif mode == "velocity":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        res = (
            np.array([res[0], res[1], res[2], res[3], 1e-6 * np.ones(res[0].shape)])
            if res.ndim > 2
            else np.array([res[0], res[1], min_confidence * np.ones(res[0].shape)])
        )
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = np.argmax(res, axis=0)
        o_, o = res == 0, res == 1
        t = tau * o + (tau_ + t_) * o_
        if time_connectivities:
            if time_connectivities is True:
                time_connectivities = connectivities
            t = time_connectivities.dot(t)
            o = (res < 2) * (t < t_)
            o_ = (res < 2) * (t >= t_)
            tau, alpha, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
            ut, st = SplicingDynamics(
                alpha=alpha, beta=beta, gamma=gamma, initial_state=[u0, s0]
            ).get_solution(tau, stacked=False)
            ut_, st_ = ut, st

        ut = ut * o + ut_ * o_
        st = st * o + st_ * o_
        alpha = alpha * o

        vt = ut * beta - st * gamma  # ds/dt
        wt = (alpha - beta * ut) * scaling  # du/dt

        vt, wt = np.clip(vt, -s, None), np.clip(wt, -u * scaling, None)
        if vt.ndim == 1:
            vt = vt.reshape(len(vt), 1)
            wt = wt.reshape(len(wt), 1)

        res = [vt, wt]

    elif mode == "velocity_residuals":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        res = (
            np.array([res[0], res[1], res[2], res[3], 1e-6 * np.ones(res[0].shape)])
            if res.ndim > 2
            else np.array([res[0], res[1], min_confidence * np.ones(res[0].shape)])
        )
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = np.argmax(res, axis=0)
        o_, o = res == 0, res == 1
        t = tau * o + (tau_ + t_) * o_
        if time_connectivities:
            if time_connectivities is True:
                time_connectivities = connectivities
            t = time_connectivities.dot(t)
            o = (res < 2) * (t < t_)
            o_ = (res < 2) * (t >= t_)
            tau, alpha, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
            ut, st = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
                tau, stacked=False
            )
            ut_, st_ = ut, st

        alpha = alpha * o

        vt = u * beta - s * gamma  # ds/dt
        wt = (alpha - beta * u) * scaling  # du/dt

        vt, wt = np.clip(vt, -s, None), np.clip(wt, -u * scaling, None)

        res = [vt, wt]

    elif mode == "soft_velocity":
        res = 1 / (2 * np.pi * np.sqrt(varx)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        o_, o = res[0], res[1]
        ut = ut * o + ut_ * o_
        st = st * o + st_ * o_
        alpha = alpha * o
        vt = ut * beta - st * gamma  # ds/dt
        wt = (alpha - beta * ut) * scaling  # du/dt
        res = [vt, wt]

    return res


def assign_timepoints(**kwargs):
    """Assign timepoints wrapper."""
    return compute_divergence(**kwargs, mode="assign_timepoints")


def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0, sorted=False):
    """Vectorize parameters across time."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)
    s0_ = spliced(t_, s0, u0, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, s0


def curve_dists(
    u,
    s,
    alpha,
    beta,
    gamma,
    t_=None,
    u0_=None,
    s0_=None,
    std_u=1,
    std_s=1,
    scaling=1,
    num=None,
):
    """Compute distances from observations to model curves."""
    if u0_ is None or s0_ is None:
        u0_, s0_ = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
            t_, stacked=False
        )

    x_obs = np.vstack([u, s]).T
    std_x = np.vstack([std_u / scaling, std_s]).T
    t0 = tau_inv(np.min(u[s > 0]), u0=u0_, alpha=0, beta=beta)

    num = np.clip(int(len(u) / 10), 50, 200) if num is None else num
    tpoints = np.linspace(0, t_, num=num)
    tpoints_ = np.linspace(0, t0, num=num)[1:]

    curve_t = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma).get_solution(
        tpoints
    )
    curve_t_ = SplicingDynamics(
        alpha=0, beta=beta, gamma=gamma, initial_state=[u0_, s0_]
    ).get_solution(tpoints_)

    # match each curve point to nearest observation
    dist, dist_ = np.zeros(len(curve_t)), np.zeros(len(curve_t_))
    for i, ci in enumerate(curve_t):
        dist[i] = np.min(np.sum((x_obs - ci) ** 2 / std_x**2, 1))
    for i, ci in enumerate(curve_t_):
        dist_[i] = np.min(np.sum((x_obs - ci) ** 2 / std_x**2, 1))

    return dist, dist_


def get_divergence(
    adata,
    mode="gene_likelihood",
    use_connectivities=True,
    clusters=None,
):
    """Get divergence for all genes in adata.

    This is a helper function to compute divergence metrics for velocity genes.
    """
    if "fit_alpha" not in adata.var.keys():
        raise ValueError("run recover_dynamics first")

    idx = ~np.isnan(adata.var["fit_alpha"])
    var_names = adata.var_names[idx]

    alpha = adata.var["fit_alpha"][idx].values
    beta = adata.var["fit_beta"][idx].values
    gamma = adata.var["fit_gamma"][idx].values
    scaling = adata.var["fit_scaling"][idx].values
    t_ = adata.var["fit_t_"][idx].values

    use_raw = adata.uns.get("recover_dynamics", {}).get("use_raw", False)

    connectivities = get_connectivities(adata) if use_connectivities else None

    n_vars = len(var_names)
    if clusters is not None:
        n_clusters = len(clusters.cat.categories)
        res = np.zeros((n_clusters, n_vars))
    else:
        res = np.zeros(n_vars)

    for i, gene in enumerate(var_names):
        _layers = adata[:, gene].layers
        u = _layers["unspliced"] if use_raw else _layers["Mu"]
        s = _layers["spliced"] if use_raw else _layers["Ms"]
        u, s = make_dense(u), make_dense(s)

        std_u = adata.var.loc[gene, "fit_std_u"]
        std_s = adata.var.loc[gene, "fit_std_s"]

        res_i = compute_divergence(
            u, s,
            alpha[i], beta[i], gamma[i], scaling[i],
            t_=t_[i],
            std_u=std_u,
            std_s=std_s,
            mode=mode,
            connectivities=connectivities,
            clusters=clusters,
        )

        if clusters is not None:
            res[:, i] = res_i
        else:
            res[i] = res_i

    return res


"""Base Class for Dynamics Recovery"""


class BaseDynamics:
    """Base class for recovering RNA dynamics.

    This class provides the core functionality for fitting dynamical models
    to RNA velocity data using an expectation-maximization approach.
    """

    def __init__(
        self,
        adata,
        gene,
        u=None,
        s=None,
        use_raw=False,
        perc=99,
        max_iter=10,
        fit_time=True,
        fit_scaling=True,
        fit_steady_states=True,
        fit_connected_states=True,
        fit_basal_transcription=None,
        high_pars_resolution=False,
        steady_state_prior=None,
        init_vals=None,
    ):
        self.s, self.u, self.use_raw = None, None, None

        _layers = adata[:, gene].layers
        self.gene = gene
        self.use_raw = use_raw or "Ms" not in _layers.keys()

        # extract actual data
        if u is None or s is None:
            u = _layers["unspliced"] if self.use_raw else _layers["Mu"]
            s = _layers["spliced"] if self.use_raw else _layers["Ms"]
        self.s, self.u = make_dense(s), make_dense(u)

        # Basal transcription
        if fit_basal_transcription:
            self.u0, self.s0 = np.min(u), np.min(s)
            self.u -= self.u0
            self.s -= self.s0
        else:
            self.u0, self.s0 = 0, 0

        self.alpha, self.beta, self.gamma = None, None, None
        self.scaling, self.t_, self.alpha_ = None, None, None
        self.u0_, self.s0_, self.weights = None, None, None
        self.weights_outer, self.weights_upper = None, None
        self.t, self.tau, self.o, self.tau_ = None, None, None, None
        self.likelihood, self.loss, self.pars = None, None, None

        self.max_iter = max_iter
        # partition to total of 5 fitting procedures
        # (t_ and alpha, scaling, rates, t_, all together)
        self.simplex_kwargs = {
            "method": "Nelder-Mead",
            "options": {"maxiter": int(self.max_iter / 5)},
        }

        self.perc = perc
        self.recoverable = True
        try:
            self.initialize_weights()
        except Warning:
            self.recoverable = False
            logg.warn(f"Model for {self.gene} could not be instantiated.")

        self.refit_time = fit_time

        self.assignment_mode = None
        self.steady_state_ratio = None
        self.steady_state_prior = steady_state_prior

        self.fit_scaling = fit_scaling
        self.fit_steady_states = fit_steady_states
        self.fit_connected_states = fit_connected_states
        self.connectivities = (
            get_connectivities(adata)
            if self.fit_connected_states is True
            else self.fit_connected_states
        )
        self.high_pars_resolution = high_pars_resolution
        self.init_vals = init_vals

        # for differential kinetic test
        self.clusters, self.cats, self.varx, self.orth_beta = None, None, None, None
        self.diff_kinetics, self.pval_kinetics, self.pvals_kinetics = None, None, None

    def initialize_weights(self, weighted=True):
        """Initialize weights for observations."""
        nonzero_s = np.ravel(self.s > 0)
        nonzero_u = np.ravel(self.u > 0)

        weights = np.array(nonzero_s & nonzero_u, dtype=bool)
        self.recoverable = np.sum(weights) > 2

        if self.recoverable:
            if weighted:
                ub_s = np.percentile(self.s[weights], self.perc)
                ub_u = np.percentile(self.u[weights], self.perc)
                if ub_s > 0:
                    weights &= np.ravel(self.s <= ub_s)
                if ub_u > 0:
                    weights &= np.ravel(self.u <= ub_u)

            self.weights = weights
            u, s = self.u[weights], self.s[weights]
            self.std_u = np.std(u)
            self.std_s = np.std(s)

            self.weights_upper = np.array(weights)
            if np.any(weights):
                w_upper = (self.u > np.max(u) / 3) & (self.s > np.max(s) / 3)
                self.weights_upper &= w_upper

    def load_pars(self, adata, gene):
        """Load previously fitted parameters."""
        idx = adata.var_names.get_loc(gene) if isinstance(gene, str) else gene
        self.alpha = adata.var["fit_alpha"][idx]
        self.beta = adata.var["fit_beta"][idx] * adata.var["fit_scaling"][idx]
        self.gamma = adata.var["fit_gamma"][idx]
        self.scaling = adata.var["fit_scaling"][idx]
        self.t_ = adata.var["fit_t_"][idx]
        self.steady_state_ratio = self.gamma / self.beta

        if "fit_steady_u" in adata.var.keys():
            self.steady_u = adata.var["fit_steady_u"][idx]
        if "fit_steady_s" in adata.var.keys():
            self.steady_s = adata.var["fit_steady_s"][idx]
        if "fit_pval_steady" in adata.var.keys():
            self.pval_steady = adata.var["fit_pval_steady"][idx]

        self.alpha_ = 0
        self.u0_, self.s0_ = SplicingDynamics(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        ).get_solution(self.t_, stacked=False)
        self.pars = [self.alpha, self.beta, self.gamma, self.t_, self.scaling]
        self.pars = np.array(self.pars)[:, None]

        lt = "latent_time"
        t = adata.obs[lt] if lt in adata.obs.keys() else adata.layers["fit_t"][:, idx]

        if isinstance(self.refit_time, bool):
            self.t, self.tau, self.o = self.get_time_assignment(t=t)
        else:
            tkey = self.refit_time
            self.t = adata.obs[tkey].values if isinstance(tkey, str) else tkey
            self.refit_time = False
            steady_states = t == self.t_
            if np.any(steady_states):
                self.t_ = np.mean(self.t[steady_states])
            self.t, self.tau, self.o = self.get_time_assignment(t=self.t)

        self.loss = [self.get_loss()]

    def get_weights(self, weighted=None, weights_cluster=None):
        """Get weights for observations."""
        weights = (
            np.array(
                self.weights_outer
                if weighted == "outer"
                else self.weights_upper
                if weighted == "upper"
                else self.weights
            )
            if weighted
            else np.ones(len(self.weights), bool)
        )
        if weights_cluster is not None and len(weights) == len(weights_cluster):
            weights &= weights_cluster
        return weights

    def get_reads(self, scaling=None, weighted=None, weights_cluster=None):
        """Get scaled reads."""
        scaling = self.scaling if scaling is None else scaling
        u, s = self.u / scaling, self.s
        if weighted or weights_cluster is not None:
            weights = self.get_weights(
                weighted=weighted, weights_cluster=weights_cluster
            )
            u, s = u[weights], s[weights]
        return u, s

    def get_vars(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        t_=None,
        u0_=None,
        s0_=None,
    ):
        """Get model variables."""
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        scaling = self.scaling if scaling is None else scaling
        if t_ is None or t_ == 0:
            t_ = self.t_ if u0_ is None else tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)
        return alpha, beta, gamma, scaling, t_

    def get_divergence(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        t_=None,
        u0_=None,
        s0_=None,
        mode=None,
        **kwargs,
    ):
        """Compute divergence between model and observations."""
        alpha, beta, gamma, scaling, t_ = self.get_vars(
            alpha, beta, gamma, scaling, t_, u0_, s0_
        )
        u, s = self.u / scaling, self.s
        kwargs.update(
            {"t_": t_, "u0_": u0_, "s0_": s0_, "std_u": self.std_u, "std_s": self.std_s}
        )
        kwargs.update(
            {
                "mode": mode,
                "assignment_mode": self.assignment_mode,
                "connectivities": self.connectivities,
                "fit_steady_states": self.fit_steady_states,
            }
        )
        res = compute_divergence(u, s, alpha, beta, gamma, scaling, **kwargs)
        return res

    def get_time_assignment(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        t_=None,
        u0_=None,
        s0_=None,
        t=None,
        refit_time=None,
        rescale_factor=None,
        weighted=None,
        weights_cluster=None,
    ):
        """Get time assignments for observations."""
        if refit_time is None:
            refit_time = self.refit_time

        if t is not None:
            t_ = self.t_ if t_ is None else t_
            o = np.array(t < t_, dtype=int)
            tau = t * o + (t - t_) * (1 - o)
        elif refit_time:
            if rescale_factor is not None:
                alpha, beta, gamma, scaling, t_ = self.get_vars(
                    alpha, beta, gamma, scaling, t_, u0_, s0_
                )

                u0_ = self.u0_ if u0_ is None else u0_
                s0_ = self.s0_ if s0_ is None else s0_
                rescale_factor *= gamma / beta

                scaling *= rescale_factor
                beta *= rescale_factor

                u0_ /= rescale_factor
                t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)

            t, tau, o = self.get_divergence(
                alpha, beta, gamma, scaling, t_, u0_, s0_, mode="assign_timepoints"
            )
            if rescale_factor is not None:
                t *= self.t_ / t_
                tau *= self.t_ / t_
        else:
            t, tau, o = self.t, self.tau, self.o

        if weighted or weights_cluster is not None:
            weights = self.get_weights(
                weighted=weighted, weights_cluster=weights_cluster
            )
            t, tau, o = t[weights], tau[weights], o[weights]
        return t, tau, o

    def get_vals(
        self,
        t=None,
        t_=None,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        u0_=None,
        s0_=None,
        refit_time=None,
    ):
        """Get all model values."""
        alpha, beta, gamma, scaling, t_ = self.get_vars(
            alpha, beta, gamma, scaling, t_, u0_, s0_
        )
        t, tau, o = self.get_time_assignment(
            alpha, beta, gamma, scaling, t_, u0_, s0_, t, refit_time
        )
        return t, t_, alpha, beta, gamma, scaling

    def get_dists(
        self,
        t=None,
        t_=None,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        u0_=None,
        s0_=None,
        refit_time=None,
        weighted=True,
        weights_cluster=None,
        reg=None,
    ):
        """Get distances between model and observations."""
        weight_args = {"weighted": weighted, "weights_cluster": weights_cluster}
        u, s = self.get_reads(scaling, **weight_args)

        alpha, beta, gamma, scaling, t_ = self.get_vars(
            alpha, beta, gamma, scaling, t_, u0_, s0_
        )
        t, tau, o = self.get_time_assignment(
            alpha, beta, gamma, scaling, t_, u0_, s0_, t, refit_time, **weight_args
        )

        tau, alpha, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
        ut, st = SplicingDynamics(
            alpha=alpha, beta=beta, gamma=gamma, initial_state=[u0, s0]
        ).get_solution(tau, stacked=False)

        udiff = np.array(ut - u) / self.std_u * scaling
        sdiff = np.array(st - s) / self.std_s
        if reg is None:
            reg = 0
            if self.steady_state_ratio is not None:
                reg = (gamma / beta - self.steady_state_ratio) * s / self.std_s
        return udiff, sdiff, reg

    def get_residuals_linear(self, **kwargs):
        """Get linear residuals."""
        udiff, sdiff, reg = self.get_dists(**kwargs)
        return udiff, sdiff

    def get_residuals(self, **kwargs):
        """Get signed residuals."""
        udiff, sdiff, reg = self.get_dists(**kwargs)
        return np.sign(sdiff) * np.sqrt(udiff**2 + sdiff**2)

    def get_distx(self, noise_model="normal", regularize=True, **kwargs):
        """Get distance metric."""
        udiff, sdiff, reg = self.get_dists(**kwargs)
        distx = udiff**2 + sdiff**2
        if regularize:
            distx += reg**2
        return np.sqrt(distx) if noise_model == "laplace" else distx

    def get_se(self, **kwargs):
        """Get sum of squared errors."""
        return np.sum(self.get_distx(**kwargs))

    def get_mse(self, **kwargs):
        """Get mean squared error."""
        return np.mean(self.get_distx(**kwargs))

    def get_loss(
        self,
        t=None,
        t_=None,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        u0_=None,
        s0_=None,
        refit_time=None,
    ):
        """Get loss function value."""
        kwargs = {
            "t": t,
            "t_": t_,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "scaling": scaling,
        }
        kwargs.update({"u0_": u0_, "s0_": s0_, "refit_time": refit_time})
        return self.get_se(**kwargs)

    def get_loglikelihood(self, varx=None, noise_model="normal", **kwargs):
        """Get log-likelihood."""
        if "weighted" not in kwargs:
            kwargs.update({"weighted": "upper"})
        udiff, sdiff, reg = self.get_dists(**kwargs)

        distx = udiff**2 + sdiff**2 + reg**2
        eucl_distx = np.sqrt(distx)
        n = np.clip(len(distx) - len(self.u) * 0.01, 2, None)

        # compute variance / equivalent to np.var(np.sign(sdiff) * np.sqrt(distx))
        if varx is None:
            varx = np.mean(distx) - np.mean(np.sign(sdiff) * eucl_distx) ** 2
        varx += varx == 0  # edge case of mRNAs levels to be the same across all cells

        if noise_model == "normal":
            loglik = -1 / 2 / n * np.sum(distx) / varx
            loglik -= 1 / 2 * np.log(2 * np.pi * varx)
        elif noise_model == "laplace":
            loglik = -1 / np.sqrt(2) / n * np.sum(eucl_distx) / np.sqrt(varx)
            loglik -= 1 / 2 * np.log(2 * varx)
        else:
            raise ValueError("That noise model is not supported.")
        return loglik

    def get_likelihood(self, **kwargs):
        """Get likelihood."""
        if "weighted" not in kwargs:
            kwargs.update({"weighted": "upper"})
        likelihood = np.exp(self.get_loglikelihood(**kwargs))
        return likelihood

    def get_curve_likelihood(self):
        """Get likelihood based on curve distances."""
        alpha, beta, gamma, scaling, t_ = self.get_vars()
        u, s = self.get_reads(scaling, weighted=False)
        varx = self.get_variance()

        kwargs = {"std_u": self.std_u, "std_s": self.std_s, "scaling": scaling}
        dist, dist_ = curve_dists(u, s, alpha, beta, gamma, t_, **kwargs)
        log_likelihood = -0.5 / len(dist) * np.sum(dist) / varx - 0.5 * np.log(
            2 * np.pi * varx
        )
        log_likelihood_ = -0.5 / len(dist_) * np.sum(dist_) / varx - 0.5 * np.log(
            2 * np.pi * varx
        )
        likelihood = np.exp(np.max([log_likelihood, log_likelihood_]))
        return likelihood

    def get_variance(self, **kwargs):
        """Get variance."""
        if "weighted" not in kwargs:
            kwargs.update({"weighted": "upper"})
        udiff, sdiff, reg = self.get_dists(**kwargs)
        distx = udiff**2 + sdiff**2
        return np.mean(distx) - np.mean(np.sign(sdiff) * np.sqrt(distx)) ** 2

    def get_ut(self, **kwargs):
        """Get unspliced prediction."""
        t, t_, alpha, beta, gamma, scaling = self.get_vals(**kwargs)
        tau, alpha, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
        return unspliced(tau, u0, alpha, beta)

    def get_st(self, **kwargs):
        """Get spliced prediction."""
        t, t_, alpha, beta, gamma, scaling = self.get_vals(**kwargs)
        tau, alpha, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
        return spliced(tau, s0, u0, alpha, beta, gamma)

    def get_vt(self, mode="soft_eval"):
        """Get velocity prediction."""
        alpha, beta, gamma, scaling, _ = self.get_vars()
        o_, o, ut, st = self.get_divergence(mode=mode)
        return ut * beta - st * gamma
