"""pruning and merging costs, and parent neuron synthesis"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .kernels import j1, relu_self_kernel

_TINY = 1e-12

def prune_cost(capacity_i, layer_capacity, n_active):
    """J_prune = N ||f_i||/(E_a - ||f_i||)"""
    denom = max(layer_capacity - capacity_i, _TINY)
    return n_active * capacity_i / denom


@dataclass
class MergeResult:
    cost: float                 # J_merge
    s_star: float               # optimal parent magnitude = ||f_p||_H (Eq. 12)
    distortion: float           # D = sqrt(||f_i - f_p||^2 + ||f_j - f_p||^2)
    alignment: float            # b = <psi, f_i + f_j>_H after the phase check
    rho_hat: float              # warped correlation of the merged pair
    coeffs: np.ndarray          # (c1, c2): u_hat = c1 w~_i + c2 w~_j
    parent: dict = field(default_factory=dict)  # physical parameters


def merge_pair(layer, i, j, layer_capacity=None, rho=None):
    caps = layer.capacities()
    if layer_capacity is None:
        layer_capacity = float(caps.sum())
    n_active = layer.n_neurons

    w_tilde = layer.augmented_in()
    wt_i, wt_j = w_tilde[i], w_tilde[j]
    wo_i, wo_j = layer.w_out[i], layer.w_out[j]
    means = layer.preact_mean()
    stds = layer.preact_std()
    k_self = layer.self_kernels()
    rho = layer.rho_hat(i, j) if rho is None else float(np.clip(rho, -1.0, 1.0))

    #optimal direction u_hat: principal right-singular vector of
    #A = wo_i wt_i^T + wo_j wt_j^T  (Eq 14). A is rank <= 2 and its row
    #space is span{wt_i, wt_j}, so the singular problem reduces to the
    # 2x2 eigenproblem (G_out G_in) alpha = lambda alpha with
    # G_in = W~^T W~ and G_out = W_out^T W_out 
    # The direct SVD remains as a fallback and as
    # the reference implementation for the equivalence test
    g_in = np.array([[wt_i @ wt_i, wt_i @ wt_j],
                     [wt_j @ wt_i, wt_j @ wt_j]])
    g_out = np.array([[wo_i @ wo_i, wo_i @ wo_j],
                      [wo_j @ wo_i, wo_j @ wo_j]])
    evals, evecs = np.linalg.eig(g_out @ g_in)
    alpha = np.real(evecs[:, int(np.argmax(np.real(evals)))])
    u = alpha[0] * wt_i + alpha[1] * wt_j
    u_norm = np.linalg.norm(u)
    if u_norm > 1e-10:
        coeffs = alpha / u_norm                      # u_hat = c1 wt_i + c2 wt_j
    else:
        A = np.outer(wo_i, wt_i) + np.outer(wo_j, wt_j)
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        u_hat = vt[0]
        basis = np.stack([wt_i, wt_j], axis=1)
        coeffs, *_ = np.linalg.lstsq(basis, u_hat, rcond=None)

    # evaluate both polarities in the exact
    # objective ||sum_k K(u, w~_k) w_out_k||/sqrt(K(u, u)).
    best = None
    for sign in (1.0, -1.0):
        c = sign * coeffs
        stats = _combined_preact_stats(c, means[i], means[j], stds[i], stds[j], rho)
        mean_u, std_u = stats
        k_uu = relu_self_kernel(mean_u, std_u)
        if k_uu < _TINY:
            continue
        k_ui, k_uj = _cross_kernels_with_children(
            c, std_u, k_uu, stds[i], stds[j], k_self[i], k_self[j], rho)
        z = k_ui * wo_i + k_uj * wo_j
        objective = np.linalg.norm(z) / np.sqrt(k_uu)
        if best is None or objective > best["objective"]:
            best = dict(objective=objective, c=c, mean_u=mean_u, std_u=std_u,
                        k_uu=k_uu, z=z)
    if best is None:
        return MergeResult(cost=np.inf, s_star=0.0, distortion=np.inf,
                           alignment=0.0, rho_hat=rho, coeffs=coeffs)

    c = best["c"]
    k_uu = best["k_uu"]
    z = best["z"]
    v_star = z / max(np.linalg.norm(z), _TINY)
    b_val = best["objective"]                       # <psi, f_i + f_j>_H >= 0

    #optimal scale (Eq 12)
    a_val = caps[i] ** 2 + caps[j] ** 2
    e_rem = max(layer_capacity - caps[i] - caps[j], 0.0)
    s_star = (a_val + b_val * e_rem) / (2.0 * e_rem + b_val)

    #distortion and cost (6.3)
    d_sq = max(a_val + 2.0 * s_star**2 - 2.0 * s_star * b_val, 0.0)
    distortion = float(np.sqrt(d_sq))
    e_terminal = e_rem + s_star
    cost = n_active * distortion / max(e_terminal, _TINY)

    #physical parameters 
    parent = _recover_physical(layer, c, best, v_star, s_star,
                               wt_i, wt_j, wo_i, wo_j)

    return MergeResult(cost=float(cost), s_star=float(s_star),
                       distortion=distortion, alignment=float(b_val),
                       rho_hat=float(rho), coeffs=c, parent=parent)




def _combined_preact_stats(c, m_i, m_j, s_i, s_j, rho):
    mean_u = c[0] * m_i + c[1] * m_j
    var_u = c[0] ** 2 * s_i**2 + c[1] ** 2 * s_j**2 + 2.0 * c[0] * c[1] * s_i * s_j * rho
    return float(mean_u), float(np.sqrt(max(var_u, 0.0)))


def _cross_kernels_with_children(c, std_u, k_uu, s_i, s_j, k_ii, k_jj, rho):
    if std_u < _TINY:
        return 0.0, 0.0
    cov_ui = c[0] * s_i**2 + c[1] * s_i * s_j * rho
    cov_uj = c[0] * s_i * s_j * rho + c[1] * s_j**2
    rho_ui = np.clip(cov_ui / max(std_u * s_i, _TINY), -1.0, 1.0)
    rho_uj = np.clip(cov_uj / max(std_u * s_j, _TINY), -1.0, 1.0)
    k_ui = j1(rho_ui) * np.sqrt(k_uu * k_ii)
    k_uj = j1(rho_uj) * np.sqrt(k_uu * k_jj)
    return float(k_ui), float(k_uj)


def _recover_physical(layer, c, best, v_star, s_star, wt_i, wt_j, wo_i, wo_j):
    k_uu = best["k_uu"]
    r_f = np.sqrt(np.dot(wt_i, wt_i) + np.dot(wt_j, wt_j))
    r_f /= max(np.sqrt(np.dot(wo_i, wo_i) + np.dot(wo_j, wo_j)), _TINY)
    k_quarter = max(k_uu, _TINY) ** 0.25
    u_hat = c[0] * wt_i + c[1] * wt_j
    u_hat = u_hat / max(np.linalg.norm(u_hat), _TINY)
    s_in = np.sqrt(max(s_star, 0.0) * r_f) / k_quarter
    s_out = np.sqrt(max(s_star, 0.0) / max(r_f, _TINY)) / k_quarter
    w_tilde_p = s_in * u_hat
    w_eff_p, b_p = w_tilde_p[:-1], float(w_tilde_p[-1])
    w_out_p = s_out * v_star

    mean_p = s_in * best["mean_u"]
    gamma_p = s_in * best["std_u"]            
    beta_p = mean_p
    mu_p = beta_p - b_p
    eps = layer.eps
    if gamma_p**2 >= eps:
        sigma_p = np.sqrt(gamma_p**2 - eps)
        w_raw_p = w_eff_p                           
    elif gamma_p > _TINY:
        sigma_p = 0.0
        w_raw_p = (np.sqrt(eps) / gamma_p) * w_eff_p
        mu_p = (np.sqrt(eps) / gamma_p) * (beta_p - b_p)
    else:
        #realize the null operator
        sigma_p, w_raw_p, mu_p, beta_p, gamma_p = 0.0, np.zeros_like(w_eff_p), 0.0, 0.0, 0.0
        w_out_p = np.zeros_like(w_out_p)
    return dict(w_raw=w_raw_p, w_out=w_out_p, gamma=float(gamma_p),
                beta=float(beta_p), mu=float(mu_p), sigma=float(sigma_p))
