# type: ignore
"""
Utility functions for RNA velocity analysis.

This module contains helper functions used across the tools subpackage.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import gaussian_kde


def make_dense(X):
    """Convert sparse matrix to dense array if needed."""
    XA = X.toarray() if issparse(X) and X.ndim == 2 else X.A1 if issparse(X) else X
    if XA.ndim == 2:
        XA = XA[0] if XA.shape[0] == 1 else XA[:, 0] if XA.shape[1] == 1 else XA
    return np.array(XA)


def round(x, precision=2):
    """Round to specified precision."""
    return np.round(x, precision)


def make_unique_list(key, allow_array=False):
    """Make a list unique while preserving order."""
    if isinstance(key, (list, tuple, np.ndarray)):
        if allow_array:
            return list(dict.fromkeys(key))
        else:
            return [k for k in dict.fromkeys(key)]
    return key if isinstance(key, str) else [key]


def scale(X, min=0, max=1):
    """Scale array to specified range [min, max].

    This matches scVelo's original implementation which has unusual behavior:
    it shifts the minimum to 'min', then scales so the maximum becomes 'max'.
    """
    X = np.array(X, dtype=np.float64)
    idx = np.isfinite(X)
    if np.any(idx):
        X = X.copy()
        # Shift minimum to 'min' parameter
        X[idx] = X[idx] - X[idx].min() + min
        # Scale maximum to 'max' parameter
        xmax = X[idx].max()
        X[idx] = X[idx] / xmax * max if xmax != 0 else X[idx] * max
    return X


def vcorrcoef(X, y, mode="pearsons", axis=-1):
    """Pearsons/Spearmans correlation coefficients.

    Use Pearsons / Spearmans to test for linear / monotonic relationship.

    Parameters
    ----------
    X : np.ndarray
        Data vector or matrix
    y : np.ndarray
        Data vector or matrix
    mode : str (default: 'pearsons')
        Which correlation metric to use: 'pearsons' or 'spearmans'
    axis : int (default: -1)
        Axis along which to compute correlation

    Returns
    -------
    corr : array
        Correlation coefficients
    """
    if issparse(X):
        X = np.array(X.toarray())
    if issparse(y):
        y = np.array(y.toarray())
    if axis == 0:
        if X.ndim > 1:
            X = np.array(X.T)
        if y.ndim > 1:
            y = np.array(y.T)
    if X.shape[axis] != y.shape[axis]:
        X = X.T
    if mode in {"spearmans", "spearman"}:
        from scipy.stats.stats import rankdata

        X = np.apply_along_axis(rankdata, axis=-1, arr=X)
        y = np.apply_along_axis(rankdata, axis=-1, arr=y)
    Xm = np.array(X - (np.nanmean(X, -1)[:, None] if X.ndim > 1 else np.nanmean(X, -1)))
    ym = np.array(y - (np.nanmean(y, -1)[:, None] if y.ndim > 1 else np.nanmean(y, -1)))
    corr = np.nansum(Xm * ym, -1) / np.sqrt(
        np.nansum(Xm**2, -1) * np.nansum(ym**2, -1)
    )
    return corr


def test_bimodality(x, kde=True, kde_width=0.3):
    """Test for bimodality using kernel density estimation.

    Parameters
    ----------
    x : array-like
        Data to test for bimodality
    kde : bool
        Whether to use KDE for density estimation
    kde_width : float
        Bandwidth for KDE

    Returns
    -------
    score : float
        Bimodality score
    pval : float
        P-value for bimodality
    means : list
        Means of the two modes
    """
    x = np.array(x).ravel()
    x = x[np.isfinite(x)]

    if len(x) < 3:
        raise ValueError("Not enough valid samples for bimodality test")

    # Simple bimodality detection using KDE
    if kde and len(np.unique(x)) > 2:
        try:
            # Fit KDE
            kernel = gaussian_kde(x, bw_method=kde_width)

            # Evaluate on a grid
            x_min, x_max = np.min(x), np.max(x)
            x_grid = np.linspace(x_min, x_max, 100)
            density = kernel(x_grid)

            # Find local maxima
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(density)

            if len(peaks) >= 2:
                # Get the two highest peaks
                peak_heights = density[peaks]
                top_peaks = peaks[np.argsort(peak_heights)[-2:]]

                # Get means at peak locations
                means = [x_grid[p] for p in sorted(top_peaks)]

                # Simple score based on separation
                separation = abs(means[1] - means[0]) / (x_max - x_min)
                score = separation

                # Simple p-value estimate (lower is more significant)
                pval = 1.0 / (1.0 + score * 10)

                return score, pval, means
            else:
                # Not bimodal
                mean_val = np.mean(x)
                return 0.0, 1.0, [mean_val, mean_val]

        except Exception as e:
            warnings.warn(f"KDE failed: {e}, falling back to simple method")

    # Fallback: simple bimodality using median split
    median = np.median(x)
    lower = x[x <= median]
    upper = x[x > median]

    if len(lower) > 0 and len(upper) > 0:
        mean_lower = np.mean(lower)
        mean_upper = np.mean(upper)

        # Separation relative to range
        separation = (mean_upper - mean_lower) / (np.max(x) - np.min(x))
        score = separation
        pval = 1.0 / (1.0 + score * 10)

        return score, pval, [mean_lower, mean_upper]
    else:
        mean_val = np.mean(x)
        return 0.0, 1.0, [mean_val, mean_val]
