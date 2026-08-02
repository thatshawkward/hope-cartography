"""quadrature activation kernels for the Gaussian surrogate beyond PH-1"""

from __future__ import annotations
import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def gelu(x):
    """GPT 2 tanh approximate GELU"""
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))


def _gh(deg):
    """probabilists' Gauss-Hermite nodes/weights for E_{z~N(0,1)}[f(z)]"""
    z, w = np.polynomial.hermite_e.hermegauss(deg)
    return z, w / w.sum()


def act_moments(beta, gamma, act=gelu, deg=96):
    """E[act(y)] and E[act(y)^2] for y ~ N(beta, gamma^2) vectorized"""
    beta = np.atleast_1d(np.asarray(beta, float))
    gamma = np.atleast_1d(np.asarray(gamma, float))
    z, w = _gh(deg)
    y = gamma[:, None] * z[None, :] + beta[:, None]
    a = act(y)
    return a @ w, (a * a) @ w


def act_self_kernel(beta, gamma, act=gelu, deg=96):
    """K_ii = E[act(y_i)^2] under the surrogate"""
    return act_moments(beta, gamma, act=act, deg=deg)[1]


def act_pair_kernel(bi, gi, bj, gj, rho, act=gelu, deg=48):
    """K_ij = E[act(y_i) act(y_j)] correlated Gaussians batched over pairs"""
    bi, gi = np.atleast_1d(np.asarray(bi, float)), np.atleast_1d(np.asarray(gi, float))
    bj, gj = np.atleast_1d(np.asarray(bj, float)), np.atleast_1d(np.asarray(gj, float))
    rho = np.clip(np.atleast_1d(np.asarray(rho, float)), -0.999, 0.999)
    z, w = _gh(deg)
    z1 = z[None, :, None]
    z2 = (rho[:, None, None] * z1 +
          np.sqrt(1.0 - rho[:, None, None] ** 2) * z[None, None, :])
    yi = gi[:, None, None] * z1 + bi[:, None, None]
    yj = gj[:, None, None] * z2 + bj[:, None, None]
    vals = act(yi) * act(yj)
    return np.einsum("pab,a,b->p", vals, w, w)


def act_capacity(w_out_norm, beta, gamma, act=gelu, deg=96):
    """||f_i|| = ||w_out_i|| sqrt(K_ii) under the surrogate"""
    k = act_self_kernel(beta, gamma, act=act, deg=deg)
    return np.asarray(w_out_norm, float) * np.sqrt(np.maximum(k, 0.0))
