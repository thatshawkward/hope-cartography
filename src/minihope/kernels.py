"""Closed form ReLU kernels for the HOPE framework"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

_TINY = 1e-12


def relu_self_kernel(mean, std):
    """K(i, i) = E[ReLU(y)^2] for y ~ N(mean, std^2)"""
    mean = np.asarray(mean, dtype=float)
    std = np.abs(np.asarray(std, dtype=float))
    safe_std = np.maximum(std, _TINY)
    c = mean / safe_std
    biased = (safe_std**2 + mean**2) * norm.cdf(c) + mean * safe_std * norm.pdf(c)
    out = np.where(std < _TINY, np.maximum(mean, 0.0) ** 2, biased)
    return float(out) if out.ndim == 0 else out


def j1(rho):
    """
    j1(rho) = (sqrt(1 - rho^2) + (pi - arccos(rho)) * rho)/pi
    so that K(i, j) ~= j1(rho_hat) * sqrt(K(i,i) K(j,j))  (Eq 5)
    Endpoints: j1(1) = 1, j1(0) = 1/pi, j1(-1) = 0
    """
    rho = np.clip(np.asarray(rho, dtype=float), -1.0, 1.0)
    out = (np.sqrt(1.0 - rho**2) + (np.pi - np.arccos(rho)) * rho) / np.pi
    return float(out) if out.ndim == 0 else out


def warped_correlation(rho_eff, r_i, r_j):
    rho_eff = float(np.clip(rho_eff, -1.0, 1.0))
    if abs(rho_eff) >= 1.0 - 1e-9:
        return float(np.sign(rho_eff))
    kappa = rho_eff / (1.0 - rho_eff**2) * float(r_i) * float(r_j)
    return float(2.0 * kappa / (1.0 + np.sqrt(1.0 + 4.0 * kappa**2)))


def relu_cross_kernel(k_ii, k_jj, rho_hat):
    return j1(rho_hat) * np.sqrt(k_ii * k_jj)
