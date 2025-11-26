# type: ignore
"""
DynamicsRecovery class for fitting dynamical models to individual genes.

This module implements the EM-based parameter estimation for RNA velocity
dynamics recovery on a per-gene basis.
"""

import numpy as np
from scipy.optimize import minimize

from scvelo import logging as logg

from ._em_model_utils import BaseDynamics, convolve, linreg, tau_inv, unspliced
from .utils import test_bimodality


class DynamicsRecovery(BaseDynamics):
    """Dynamics recovery with EM optimization for a single gene."""

    def __init__(self, adata, gene, load_pars=None, **kwargs):
        super().__init__(adata, gene, **kwargs)
        if load_pars and "fit_alpha" in adata.var.keys():
            self.load_pars(adata, gene)
        elif self.recoverable:
            self.initialize()

    def initialize(self):
        """Initialize parameters for EM algorithm."""
        # set weights
        u, s, w, perc = self.u, self.s, self.weights, 98
        u_w = u[w]
        s_w = s[w]

        # initialize scaling
        self.std_u, self.std_s = np.std(u_w), np.std(s_w)
        if self.std_u == 0 or self.std_s == 0:
            self.std_u = self.std_s = 1
        _scaling = self.fit_scaling
        if isinstance(_scaling, bool) and _scaling:
            scaling = self.std_u / self.std_s
        elif isinstance(_scaling, bool):
            scaling = 1
        else:
            scaling = _scaling

        u, u_w = u / scaling, u_w / scaling

        # initialize beta and gamma from extreme quantiles of s
        weights_s = s_w >= np.percentile(s_w, perc, axis=0)
        weights_u = u_w >= np.percentile(u_w, perc, axis=0)

        _prior = self.steady_state_prior
        weights_g = weights_s if _prior is None else weights_s | _prior[w]
        beta = 1
        gamma = linreg(convolve(u_w, weights_g), convolve(s_w, weights_g)) + 1e-6

        # adjust gamma / beta * scaling clipped to adapt faster to extreme ratios
        if gamma < 0.05 / scaling:
            gamma *= 1.2
        elif gamma > 1.5 / scaling:
            gamma /= 1.2

        u_inf, s_inf = u_w[weights_u | weights_s].mean(), s_w[weights_s].mean()
        u0_, s0_ = u_inf, s_inf
        alpha = u_inf * beta

        if self.init_vals is not None:  # just to test different EM start
            self.init_vals = np.array(self.init_vals)
            alpha, beta, gamma = np.array([alpha, beta, gamma]) * self.init_vals

        # initialize switching from u quantiles and alpha from s quantiles
        try:
            _, pval_u, means_u = test_bimodality(u_w, kde=True)
            _, pval_s, means_s = test_bimodality(s_w, kde=True)
        except ValueError as e:
            logg.warn(f"skipping bimodality check for {self.gene}: {e}.")
            _, _, pval_u, pval_s = 0, 0, 1, 1
            means_u, means_s = [0, 0], [0, 0]

        self.pval_steady = max(pval_u, pval_s)
        self.steady_u = means_u[1]
        self.steady_s = means_s[1]

        if self.pval_steady < 1e-3:
            u_inf = np.mean([u_inf, self.steady_u])
            alpha = gamma * s_inf
            beta = alpha / u_inf
            u0_, s0_ = u_inf, s_inf

        t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)

        # update object with initialized vars
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.scaling, self.alpha_ = scaling, 0
        self.u0_, self.s0_, self.t_ = u0_, s0_, t_
        self.pars = np.array([alpha, beta, gamma, self.t_, self.scaling])[:, None]

        # initialize time point assignment
        self.t, self.tau, self.o = self.get_time_assignment()
        self.loss = [self.get_loss()]

        if self.fit_scaling:
            self.initialize_scaling(sight=0.5)
            self.initialize_scaling(sight=0.1)

        self.steady_state_ratio = self.gamma / self.beta

        self.set_callbacks()

    def initialize_scaling(self, sight=0.5):
        """Fit scaling and update if improved."""
        z_vals = self.scaling + np.linspace(-1, 1, num=4) * self.scaling * sight
        for z in z_vals:
            self.update(scaling=z, beta=self.beta / self.scaling * z)

    def fit(self, assignment_mode=None):
        """Main fitting procedure."""
        if self.max_iter > 0:
            # for comparison with exact time assignment
            if assignment_mode == "full_projection":
                self.assignment_mode = assignment_mode

            # pre-train with explicit time assignment
            self.fit_t_and_alpha()
            if self.fit_scaling:
                self.fit_scaling_()
            self.fit_rates()
            self.fit_t_()

            # actual EM (each iteration of simplex downhill)
            self.fit_t_and_rates()

            # train with optimal time assignment (oth. projection)
            self.assignment_mode = assignment_mode
            self.update(adjust_t_=False)
            self.fit_t_and_rates(refit_time=False)

        self.update()
        self.tau, self.tau_ = self.get_divergence(mode="tau")
        self.likelihood = self.get_likelihood(refit_time=False)
        self.varx = self.get_variance()

    def fit_alpha(self, sight=0.5, **kwargs):
        """Fit alpha parameter."""
        def mse(x):
            return self.get_mse(alpha=x[0], **kwargs)

        val = self.alpha
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(alpha=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(alpha=res.x[0])

    def fit_beta(self, sight=0.5, **kwargs):
        """Fit beta parameter."""
        def mse(x):
            return self.get_mse(beta=x[0], **kwargs)

        val = self.beta
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(beta=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(beta=res.x[0])

    def fit_gamma(self, sight=0.5, **kwargs):
        """Fit gamma parameter."""
        def mse(x):
            return self.get_mse(gamma=x[0], **kwargs)

        val = self.gamma
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(gamma=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(gamma=res.x[0])

    def fit_t_and_alpha(self, **kwargs):
        """Fit t_ and alpha together."""
        def mse(x):
            return self.get_mse(t_=x[0], alpha=x[1], **kwargs)

        alpha_vals = self.alpha + np.linspace(-1, 1, num=5) * self.alpha / 10
        for alpha in alpha_vals:
            self.update(alpha=alpha)
        x0, cb = np.array([self.t_, self.alpha]), self.cb_fit_t_and_alpha
        res = minimize(mse, x0, callback=cb, **self.simplex_kwargs)
        self.update(t_=res.x[0], alpha=res.x[1])

    def fit_rates(self, **kwargs):
        """Fit alpha and gamma rates."""
        def mse(x):
            return self.get_mse(alpha=x[0], gamma=x[1], **kwargs)

        x0, cb = np.array([self.alpha, self.gamma]), self.cb_fit_rates
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(alpha=res.x[0], gamma=res.x[1])

    def fit_t_(self, **kwargs):
        """Fit switching time t_."""
        def mse(x):
            return self.get_mse(t_=x[0], **kwargs)

        res = minimize(mse, self.t_, callback=self.cb_fit_t_, **self.simplex_kwargs)
        self.update(t_=res.x[0])

    def fit_rates_all(self, **kwargs):
        """Fit all rate parameters."""
        def mse(x):
            return self.get_mse(alpha=x[0], beta=x[1], gamma=x[2], **kwargs)

        x0, cb = np.array([self.alpha, self.beta, self.gamma]), self.cb_fit_rates_all
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(alpha=res.x[0], beta=res.x[1], gamma=res.x[2])

    def fit_t_and_rates(self, **kwargs):
        """Fit all parameters together."""
        def mse(x):
            return self.get_mse(t_=x[0], alpha=x[1], beta=x[2], gamma=x[3], **kwargs)

        x0 = np.array([self.t_, self.alpha, self.beta, self.gamma])
        cb = self.cb_fit_t_and_rates
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(t_=res.x[0], alpha=res.x[1], beta=res.x[2], gamma=res.x[3])

    def fit_scaling_(self, **kwargs):
        """Fit scaling parameter."""
        def mse(x):
            return self.get_mse(t_=x[0], beta=x[1], scaling=x[2], **kwargs)

        x0, cb = np.array([self.t_, self.beta, self.scaling]), self.cb_fit_scaling_
        res = minimize(mse, x0, callback=cb, **self.simplex_kwargs)
        self.update(t_=res.x[0], beta=res.x[1], scaling=res.x[2])

    # Callback functions for the Optimizer
    def cb_fit_t_and_alpha(self, x):
        """Callback for fit_t_and_alpha."""
        self.update(t_=x[0], alpha=x[1])

    def cb_fit_scaling_(self, x):
        """Callback for fit_scaling."""
        self.update(t_=x[0], beta=x[1], scaling=x[2])

    def cb_fit_rates(self, x):
        """Callback for fit_rates."""
        self.update(alpha=x[0], gamma=x[1])

    def cb_fit_t_(self, x):
        """Callback for fit_t_."""
        self.update(t_=x[0])

    def cb_fit_t_and_rates(self, x):
        """Callback for fit_t_and_rates."""
        self.update(t_=x[0], alpha=x[1], beta=x[2], gamma=x[3])

    def cb_fit_rates_all(self, x):
        """Callback for fit_rates_all."""
        self.update(alpha=x[0], beta=x[1], gamma=x[2])

    def set_callbacks(self):
        """Set or disable callbacks based on resolution setting."""
        # Overwrite callbacks
        if not self.high_pars_resolution:
            self.cb_fit_t_and_alpha = None
            self.cb_fit_scaling_ = None
            self.cb_fit_rates = None
            self.cb_fit_t_ = None
            self.cb_fit_t_and_rates = None
            self.cb_fit_rates_all = None

    def update(
        self,
        t=None,
        t_=None,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        u0_=None,
        s0_=None,
        adjust_t_=True,
    ):
        """Update parameters if loss improves."""
        loss_prev = self.loss[-1] if len(self.loss) > 0 else 1e6

        _vars = self.get_vars(alpha, beta, gamma, scaling, t_, u0_, s0_)
        _time = self.get_time_assignment(alpha, beta, gamma, scaling, t_, u0_, s0_, t)
        alpha, beta, gamma, scaling, t_ = _vars
        t, tau, o = _time
        loss = self.get_loss(t, t_, alpha, beta, gamma, scaling)
        perform_update = loss < loss_prev

        on = self.o == 1
        if adjust_t_ and np.any(on):
            if not perform_update:
                alpha, beta, gamma, scaling, t_ = self.get_vars()
                t, tau, o = self.get_time_assignment()
                loss = self.get_loss()

            alt_t_ = t[on].max()
            if 0 < alt_t_ < t_:
                alt_t_ += np.max(t) / len(t) * np.sum(t == t_)
                _time = self.get_time_assignment(alpha, beta, gamma, scaling, alt_t_)
                alt_t, alt_tau, alt_o = _time
                alt_loss = self.get_loss(alt_t, alt_t_, alpha, beta, gamma, scaling)
                ut_cur = unspliced(t_, 0, alpha, beta)
                ut_alt = unspliced(alt_t_, 0, alpha, beta)

                min_loss = np.min([loss, loss_prev])
                if alt_loss * 0.99 <= min_loss or ut_cur * 0.99 < ut_alt:
                    t, tau, o, t_, loss = alt_t, alt_tau, alt_o, alt_t_, alt_loss
                    perform_update = True

        if perform_update:
            if scaling is not None:
                self.steady_u *= self.scaling / scaling
                self.u0_ *= self.scaling / scaling
            if u0_ is not None:
                self.u0_ = u0_
            if s0_ is not None:
                self.s0_ = s0_

            self.t, self.tau, self.o = t, tau, o
            self.alpha, self.beta, self.gamma = alpha, beta, gamma
            self.scaling, self.t_ = scaling, t_
            new_pars = np.array([alpha, beta, gamma, t_, scaling])[:, None]
            self.pars = np.c_[self.pars, new_pars]
            self.loss.append(loss)

        return perform_update
