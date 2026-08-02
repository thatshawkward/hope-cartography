"""A BatchNorm + ReLU layer viewed through HOPE"""

from __future__ import annotations
import numpy as np
from .kernels import relu_self_kernel, warped_correlation

_TINY = 1e-12


class HopeLayer:

    def __init__(self, w_raw, w_out, gamma, beta, mu, sigma, eps=1e-5):
        self.w_raw = np.atleast_2d(np.asarray(w_raw, dtype=float))
        self.w_out = np.atleast_2d(np.asarray(w_out, dtype=float))
        self.gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
        self.beta = np.atleast_1d(np.asarray(beta, dtype=float))
        self.mu = np.atleast_1d(np.asarray(mu, dtype=float))
        self.sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        self.eps = float(eps)
        n = self.w_raw.shape[0]
        if not (self.w_out.shape[0] == self.gamma.shape[0] == self.beta.shape[0]
                == self.mu.shape[0] == self.sigma.shape[0] == n):
            raise ValueError("inconsistent neuron counts across parameters")

    @property
    def n_neurons(self):
        return self.w_raw.shape[0]

    @property
    def n_in(self):
        return self.w_raw.shape[1]

    @property
    def n_out(self):
        return self.w_out.shape[1]


    def bn_scale(self):
        """gamma/sqrt(sigma^2 + eps)"""
        return self.gamma / np.sqrt(self.sigma**2 + self.eps)

    def effective_in(self):
        """w_eff = bn_scale * w_raw"""
        return self.bn_scale()[:, None] * self.w_raw

    def effective_bias(self):
        """b = beta - bn_scale * mu"""
        return self.beta - self.bn_scale() * self.mu

    def augmented_in(self):
        """w_tilde = [w_eff, b]"""
        return np.concatenate([self.effective_in(), self.effective_bias()[:, None]], axis=1)


    def preact_mean(self):
        """E[y_i] = beta_i under the surrogate"""
        return self.beta.copy()

    def preact_std(self):
        """std[y_i] = |gamma_i| sigma_i/sqrt(sigma_i^2 + eps)"""
        return np.abs(self.gamma) * self.sigma / np.sqrt(self.sigma**2 + self.eps)

    def self_kernels(self):
        return relu_self_kernel(self.preact_mean(), self.preact_std())

    def capacities(self):
        """||f_i||_H = ||w_out_i|| sqrt(K(i, i))"""
        return np.linalg.norm(self.w_out, axis=1) * np.sqrt(self.self_kernels())

    def variance_ratio(self):
        """r_i = sigma_i / ||w_raw_i||"""
        wnorm = np.maximum(np.linalg.norm(self.w_raw, axis=1), _TINY)
        return self.sigma / wnorm

    def rho_hat(self, i, j):
        w = self.effective_in()
        ni = np.linalg.norm(w[i])
        nj = np.linalg.norm(w[j])
        if ni < _TINY or nj < _TINY:
            return 0.0
        rho_eff = float(np.dot(w[i], w[j]) / (ni * nj))
        r = self.variance_ratio()
        return warped_correlation(rho_eff, r[i], r[j])

    def subset(self, indices):
        idx = np.asarray(indices, dtype=int)
        return HopeLayer(self.w_raw[idx], self.w_out[idx], self.gamma[idx],
                         self.beta[idx], self.mu[idx], self.sigma[idx], self.eps)

    def with_neuron_replaced(self, i, params):
        w_raw = self.w_raw.copy(); w_raw[i] = params["w_raw"]
        w_out = self.w_out.copy(); w_out[i] = params["w_out"]
        gamma = self.gamma.copy(); gamma[i] = params["gamma"]
        beta = self.beta.copy(); beta[i] = params["beta"]
        mu = self.mu.copy(); mu[i] = params["mu"]
        sigma = self.sigma.copy(); sigma[i] = params["sigma"]
        return HopeLayer(w_raw, w_out, gamma, beta, mu, sigma, self.eps)
