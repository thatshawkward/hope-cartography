"""Every closed form is checked against a Monte Carlo estimate of the integral
it claims to solve, and the paper's two headline invariances are asserted
"""

import numpy as np
import pytest

from minihope.kernels import j1, relu_cross_kernel, relu_self_kernel, warped_correlation
from minihope.layer import HopeLayer
from minihope.merge import merge_pair, prune_cost

RNG = np.random.default_rng(7)


#kernels

def test_self_kernel_known_value():
    # Zero bias: E[ReLU(y)^2] = gamma^2/2 exactly.
    assert relu_self_kernel(0.0, 1.7) == pytest.approx(1.7**2 / 2, rel=1e-12)
    # Degenerate std: deterministic ReLU(mean)^2.
    assert relu_self_kernel(0.8, 0.0) == pytest.approx(0.64, rel=1e-12)
    assert relu_self_kernel(-0.8, 0.0) == 0.0


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.6, 0.9), (-0.7, 1.3), (2.0, 0.4)])
def test_self_kernel_monte_carlo(mean, std):
    y = RNG.normal(mean, std, size=2_000_000)
    mc = np.mean(np.maximum(y, 0.0) ** 2)
    assert relu_self_kernel(mean, std) == pytest.approx(mc, rel=5e-3)


def test_j1_endpoints():
    assert j1(1.0) == pytest.approx(1.0)
    assert j1(0.0) == pytest.approx(1.0 / np.pi)
    assert j1(-1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.4, 0.9])
def test_cross_kernel_monte_carlo_zero_bias(rho):
    s_i, s_j = 1.4, 0.6
    cov = np.array([[s_i**2, rho * s_i * s_j], [rho * s_i * s_j, s_j**2]])
    y = RNG.multivariate_normal([0.0, 0.0], cov, size=2_000_000)
    mc = np.mean(np.maximum(y[:, 0], 0.0) * np.maximum(y[:, 1], 0.0))
    k_ii = relu_self_kernel(0.0, s_i)
    k_jj = relu_self_kernel(0.0, s_j)
    assert relu_cross_kernel(k_ii, k_jj, rho) == pytest.approx(mc, rel=1e-2, abs=1e-4)


def test_warped_correlation_isotropic_identity():
    #When BN statistics match what isotropic data would produce
    # (sigma = ||w_raw||), the warp is the identity: rho_hat == rho_eff.
    for _ in range(20):
        rho = RNG.uniform(-0.95, 0.95)
        assert warped_correlation(rho, 1.0, 1.0) == pytest.approx(rho, abs=1e-9)


def test_warped_correlation_defining_equation():
    # rho_hat solves rho/(1-rho^2) = kappa.
    for _ in range(20):
        rho_eff = RNG.uniform(-0.9, 0.9)
        r_i, r_j = RNG.uniform(0.3, 2.0, size=2)
        kappa = rho_eff / (1 - rho_eff**2) * r_i * r_j
        rh = warped_correlation(rho_eff, r_i, r_j)
        assert rh / (1 - rh**2) == pytest.approx(kappa, rel=1e-9, abs=1e-9)


#invariances

def _random_layer(n_neurons=6, n_in=10, n_out=5, zero_bias=False, seed=0):
    rng = np.random.default_rng(seed)
    w_raw = rng.normal(size=(n_neurons, n_in))
    w_out = rng.normal(size=(n_neurons, n_out))
    gamma = rng.uniform(0.5, 1.5, size=n_neurons) * rng.choice([-1, 1], size=n_neurons)
    mu = rng.normal(scale=0.5, size=n_neurons)
    sigma = rng.uniform(0.5, 2.0, size=n_neurons)
    if zero_bias:
        beta = np.zeros(n_neurons)
        mu = np.zeros(n_neurons)
    else:
        beta = rng.normal(scale=0.5, size=n_neurons)
    return HopeLayer(w_raw, w_out, gamma, beta, mu, sigma, eps=1e-5)


def test_normalization_invariance():
    # Scaling w_raw by lambda scales the BN statistics (mu, sigma) by lambda,
    # and the neuron's capacity must not move (Section 3.1).
    layer = _random_layer(seed=1)
    lam = 7.3
    scaled = HopeLayer(lam * layer.w_raw, layer.w_out, layer.gamma, layer.beta,
                       lam * layer.mu, lam * layer.sigma, layer.eps)
    # The BN stability constant makes this exact only as eps -> 0; the
    # residual is O(eps/sigma^2) ~ 1e-5 here, so allow 1e-4 relative.
    assert scaled.capacities() == pytest.approx(layer.capacities(), rel=1e-4)
    # With eps = 0 the invariance is exact to machine precision.
    exact = HopeLayer(layer.w_raw, layer.w_out, layer.gamma, layer.beta,
                      layer.mu, layer.sigma, eps=0.0)
    exact_scaled = HopeLayer(lam * layer.w_raw, layer.w_out, layer.gamma,
                             layer.beta, lam * layer.mu, lam * layer.sigma, eps=0.0)
    assert exact_scaled.capacities() == pytest.approx(exact.capacities(), rel=1e-12)


def test_resharding_invariance():
    # PH-1 symmetry: (w_eff, b) -> lambda (w_eff, b), w_out -> w_out/lambda
    # leaves ||f||_H unchanged (S5). In physical parameters this is
    # gamma -> lambda gamma, beta -> lambda beta, w_out -> w_out/lambda.
    layer = _random_layer(seed=2)
    lam = 3.1
    scaled = HopeLayer(layer.w_raw, layer.w_out / lam, lam * layer.gamma,
                       lam * layer.beta, layer.mu, layer.sigma, layer.eps)
    assert scaled.capacities() == pytest.approx(layer.capacities(), rel=1e-9)


#costs

def test_prune_cost_properties():
    assert prune_cost(1.0, 10.0, 5) == pytest.approx(5.0 / 9.0)
    assert prune_cost(1.0, 1.0 + 1e-9, 5) > 1e8
    assert prune_cost(0.0, 10.0, 5) == 0.0


def test_merge_identical_neurons_is_free():
    base = _random_layer(n_neurons=1, seed=3)
    dup = HopeLayer(np.vstack([base.w_raw, base.w_raw]),
                    np.vstack([base.w_out, base.w_out]),
                    np.repeat(base.gamma, 2), np.repeat(base.beta, 2),
                    np.repeat(base.mu, 2), np.repeat(base.sigma, 2), base.eps)
    filler = _random_layer(n_neurons=4, seed=4)
    layer = HopeLayer(np.vstack([dup.w_raw, filler.w_raw]),
                      np.vstack([dup.w_out, filler.w_out]),
                      np.concatenate([dup.gamma, filler.gamma]),
                      np.concatenate([dup.beta, filler.beta]),
                      np.concatenate([dup.mu, filler.mu]),
                      np.concatenate([dup.sigma, filler.sigma]), base.eps)
    caps = layer.capacities()
    res = merge_pair(layer, 0, 1)
    assert res.rho_hat == pytest.approx(1.0, abs=1e-6)
    assert res.distortion == pytest.approx(0.0, abs=1e-6 * caps[0])
    assert res.cost == pytest.approx(0.0, abs=1e-5)
    assert res.s_star == pytest.approx(caps[0], rel=1e-6)


def test_merge_parent_capacity_consistency():
    #the recovered physical parent pushed back through the BN forward pass
    #must reproduce the Hilbertspace magnitude ||f_p||_H == s_star.
    layer = _random_layer(n_neurons=5, seed=5)
    res = merge_pair(layer, 0, 1)
    one = HopeLayer(res.parent["w_raw"][None, :], res.parent["w_out"][None, :],
                    [res.parent["gamma"]], [res.parent["beta"]],
                    [res.parent["mu"]], [res.parent["sigma"]], layer.eps)
    assert float(one.capacities()[0]) == pytest.approx(res.s_star, rel=1e-3)


def test_merge_distortion_matches_monte_carlo_zero_bias():
    #with zero biases the zero-bias cross-kernel is exact, so the closed form
    #distortion D^2 must match a Monte Carlo estimate of
    # E||f_i - f_p||^2 + E||f_j - f_p||^2 under the pairwise surrogate.
    layer = _random_layer(n_neurons=2, zero_bias=True, seed=6)
    layer.w_raw[1] = 0.8 * layer.w_raw[0] + 0.3 * layer.w_raw[1]
    layer.sigma[:] = np.linalg.norm(layer.w_raw, axis=1)  #isotropic data
    res = merge_pair(layer, 0, 1)

    stds = layer.preact_std()
    rho = layer.rho_hat(0, 1)
    cov = np.array([[stds[0] ** 2, rho * stds[0] * stds[1]],
                    [rho * stds[0] * stds[1], stds[1] ** 2]])
    y = RNG.multivariate_normal([0.0, 0.0], cov, size=2_000_000)

    #y_p under the surrogate: the parent's preactivation is the same linear
    #combination of the children's preactivations that built u_hat, scaled
    #by s_in (App D). Equivalently, sample from N(0, gamma_p^2) jointly
    #here I reconstruct it from the coefficients
    c = res.coeffs
    w_tilde = layer.augmented_in()
    u = c[0] * w_tilde[0] + c[1] * w_tilde[1]
    #zero biases, ||w_eff_p|| = s_in exactly, and
    #y_p = s_in * (c1 y_i + c2 y_j)/||u||
    s_in = np.linalg.norm(res.parent["w_raw"])
    y_p = s_in * (c[0] * y[:, 0] + c[1] * y[:, 1]) / np.linalg.norm(u)

    f_i = np.maximum(y[:, 0], 0.0)[:, None] * layer.w_out[0]
    f_j = np.maximum(y[:, 1], 0.0)[:, None] * layer.w_out[1]
    f_p = np.maximum(y_p, 0.0)[:, None] * res.parent["w_out"]
    mc = np.mean(np.sum((f_i - f_p) ** 2, axis=1)) \
        + np.mean(np.sum((f_j - f_p) ** 2, axis=1))
    assert res.distortion**2 == pytest.approx(mc, rel=3e-2)


def test_merge_beats_prune_for_redundant_pair():
    #For a highly correlated pair, merging must be cheaper than pruning either neuron
    layer = _random_layer(n_neurons=6, seed=8)
    layer.w_raw[1] = layer.w_raw[0] + 0.05 * RNG.normal(size=layer.n_in)
    layer.w_out[1] = layer.w_out[0] + 0.05 * RNG.normal(size=layer.n_out)
    layer.gamma[1] = layer.gamma[0]
    layer.sigma[:] = np.linalg.norm(layer.w_raw, axis=1)
    caps = layer.capacities()
    e_a = float(caps.sum())
    res = merge_pair(layer, 0, 1, layer_capacity=e_a)
    j_p = min(prune_cost(caps[0], e_a, 6), prune_cost(caps[1], e_a, 6))
    assert res.cost < j_p


def test_fast_parent_solve_matches_svd_reference():
    # The 2x2 reduction must reproduce the explicit SVD parent up to sign,
    # and hence the same cost, scale, and distortion.
    for seed in range(6):
        layer = _random_layer(n_neurons=6, seed=100 + seed)
        res = merge_pair(layer, 0, 1)
        wt = layer.augmented_in()
        A = np.outer(layer.w_out[0], wt[0]) + np.outer(layer.w_out[1], wt[1])
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        u_ref = vt[0]
        u_fast = res.coeffs[0] * wt[0] + res.coeffs[1] * wt[1]
        u_fast = u_fast / np.linalg.norm(u_fast)
        align = abs(float(u_fast @ u_ref))
        assert align == pytest.approx(1.0, abs=1e-8)
