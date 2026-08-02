from __future__ import annotations
import numpy as np
from scipy import stats as sps
from minihope.kernels import j1

_TINY = 1e-12


def collect_preacts_f32(model, ctx, chunk=65536):
    outs = []
    for s in range(0, ctx.shape[0], chunk):
        x = model.features(ctx[s:s + chunk])
        outs.append((x @ model.W1 + model.b1).astype(np.float32))
    return np.concatenate(outs, axis=0)


#experiment A

def gaussianity_stats(h_pre, ks_subsample=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = h_pre.mean(axis=0)
    s = h_pre.std(axis=0) + _TINY
    z = (h_pre - m) / s
    skew = (z**3).mean(axis=0)
    exkurt = (z**4).mean(axis=0) - 3.0
    n = h_pre.shape[0]
    idx = rng.choice(n, size=min(ks_subsample, n), replace=False)
    ks = np.array([sps.kstest(z[idx, u], "norm").statistic
                   for u in range(h_pre.shape[1])])
    return dict(mean=m, std=s, skew=skew, exkurt=exkurt, ks=ks, z=z)


def gaussian_control_preacts(model, calib_x, n, seed=0):
    rng = np.random.default_rng(seed)
    mu = calib_x.mean(axis=0)
    cov = np.cov(calib_x, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
    xg = mu + rng.standard_normal((n, mu.size)) @ L.T
    return (xg @ model.W1 + model.b1).astype(np.float32)


#experiment B

def rho_hat_matrix(layer):
    w = layer.effective_in()
    norms = np.maximum(np.linalg.norm(w, axis=1), _TINY)
    c = (w / norms[:, None]) @ (w / norms[:, None]).T
    c = np.clip(c, -1.0, 1.0)
    r = layer.variance_ratio()
    near = np.abs(c) >= 1.0 - 1e-9
    c_safe = np.clip(c, -1.0 + 1e-9, 1.0 - 1e-9)
    kappa = c_safe / (1.0 - c_safe**2) * np.outer(r, r)
    rho = 2.0 * kappa / (1.0 + np.sqrt(1.0 + 4.0 * kappa**2))
    rho[near] = np.sign(c[near])
    np.fill_diagonal(rho, 1.0)
    return rho


def kernel_fidelity(layer, h_pre):
    n, h = h_pre.shape
    relu = np.maximum(h_pre, 0.0)
    emp_self = (relu.astype(np.float64)**2).mean(axis=0)
    emp_cross = (relu.T @ relu).astype(np.float64) / n
    rho_emp = np.corrcoef(h_pre.T.astype(np.float64))

    closed_self = layer.self_kernels()
    rho_mx = rho_hat_matrix(layer)
    root = np.sqrt(np.outer(closed_self, closed_self))
    closed_cross_a = j1(rho_mx) * root            # full paper pipeline
    closed_cross_b = j1(rho_emp) * root           # oracle correlation

    iu = np.triu_indices(h, k=1)
    scale = np.sqrt(np.outer(emp_self, emp_self))[iu] + _TINY
    err_a = np.abs(closed_cross_a[iu] - emp_cross[iu]) / scale
    err_b = np.abs(closed_cross_b[iu] - emp_cross[iu]) / scale
    self_rel = np.abs(closed_self - emp_self) / (emp_self + _TINY)

    cap_closed = layer.capacities()
    cap_emp = np.linalg.norm(layer.w_out, axis=1) * np.sqrt(emp_self)
    rho_cap = sps.spearmanr(cap_closed, cap_emp).statistic
    rho_rank_corr = sps.spearmanr(rho_mx[iu], rho_emp[iu]).statistic

    return dict(emp_self=emp_self, closed_self=closed_self, self_rel=self_rel,
                emp_cross=emp_cross, closed_cross_a=closed_cross_a,
                closed_cross_b=closed_cross_b, rho_mx=rho_mx, rho_emp=rho_emp,
                err_a=err_a, err_b=err_b, iu=iu,
                cap_closed=cap_closed, cap_emp=cap_emp,
                spearman_capacity=float(rho_cap),
                spearman_rho=float(rho_rank_corr))


def make_figures(outdir, char_counts, word_counts, gauss, gauss_ctrl, fid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    #zipf premise
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, counts, label in ((axes[0], char_counts, "characters"),
                              (axes[1], word_counts, "words")):
        ranks = np.arange(1, len(counts) + 1)
        ax.loglog(ranks, counts / counts.sum(), ".", ms=3)
        ax.set_xlabel("rank"); ax.set_ylabel("frequency")
        ax.set_title(f"tiny Shakespeare: {label}")
    fig.suptitle("The premise: text is Zipfian at every granularity")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1_zipf.png", dpi=140)
    plt.close(fig)

    #Gaussianity of preactivations
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    bins = np.linspace(min(gauss["exkurt"].min(), -1), gauss["exkurt"].max(), 50)
    axes[0, 0].hist(gauss["exkurt"], bins=bins, alpha=0.75, label="real text")
    axes[0, 0].hist(gauss_ctrl["exkurt"], bins=bins, alpha=0.75,
                    label="Gaussian control")
    axes[0, 0].set_xlabel("excess kurtosis"); axes[0, 0].legend()
    axes[0, 1].hist(gauss["skew"], bins=50, alpha=0.75, label="real text")
    axes[0, 1].hist(gauss_ctrl["skew"], bins=50, alpha=0.75, label="control")
    axes[0, 1].set_xlabel("skewness"); axes[0, 1].legend()
    axes[0, 2].hist(gauss["ks"], bins=50, alpha=0.75, label="real text")
    axes[0, 2].hist(gauss_ctrl["ks"], bins=50, alpha=0.75, label="control")
    axes[0, 2].set_xlabel("KS distance to normal"); axes[0, 2].legend()
    order = np.argsort(np.abs(gauss["exkurt"]))
    picks = [order[-1], order[len(order) // 2], order[0]]
    names = ["most non-Gaussian unit", "median unit", "most Gaussian unit"]
    qs = np.linspace(0.001, 0.999, 400)
    tq = sps.norm.ppf(qs)
    for ax, u, name in zip(axes[1], picks, names):
        eq = np.quantile(gauss["z"][:, u], qs)
        ax.plot(tq, eq, lw=1.2)
        ax.plot(tq, tq, "k--", lw=0.8)
        ax.set_title(f"{name} (exkurt {gauss['exkurt'][u]:+.2f})", fontsize=9)
        ax.set_xlabel("normal quantiles"); ax.set_ylabel("empirical quantiles")
    fig.suptitle("Experiment A: pre-activation shape on real text vs. the Gaussian fiction")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2_gaussianity.png", dpi=140)
    plt.close(fig)

    #Self kernels and capacities
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    lo = min(fid["emp_self"].min(), fid["closed_self"].min())
    hi = max(fid["emp_self"].max(), fid["closed_self"].max())
    axes[0].loglog([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].loglog(fid["emp_self"], fid["closed_self"], ".", ms=4, alpha=0.7)
    axes[0].set_xlabel("empirical K(i,i)"); axes[0].set_ylabel("closed-form K(i,i)")
    axes[0].set_title("self-kernels")
    axes[1].plot(np.abs(fid["_exkurt"]), fid["self_rel"], ".",
                 ms=4, alpha=0.7)
    axes[1].set_xlabel("|excess kurtosis| of unit"); axes[1].set_ylabel("relative error of K(i,i)")
    axes[1].set_title("shape error tracks non-Gaussianity")
    axes[2].plot(fid["cap_emp"], fid["cap_closed"], ".", ms=4, alpha=0.7)
    m = max(fid["cap_emp"].max(), fid["cap_closed"].max())
    axes[2].plot([0, m], [0, m], "k--", lw=0.8)
    axes[2].set_xlabel("empirical capacity"); axes[2].set_ylabel("closed-form capacity")
    axes[2].set_title(f"capacity ranking: Spearman {fid['spearman_capacity']:.4f}")
    fig.suptitle("Experiment B1: what the surrogate gets right and wrong about single neurons")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_self_capacity.png", dpi=140)
    plt.close(fig)

    #Cross kernels
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    iu = fid["iu"]
    axes[0].plot(fid["rho_emp"][iu], fid["rho_mx"][iu], ".", ms=2, alpha=0.35)
    axes[0].plot([-1, 1], [-1, 1], "k--", lw=0.8)
    axes[0].set_xlabel("empirical corr(y_i, y_j)")
    axes[0].set_ylabel("MaxEnt warped rho_hat")
    axes[0].set_title(f"correlation model: Spearman {fid['spearman_rho']:.3f}")
    axes[1].plot(fid["emp_cross"][iu], fid["closed_cross_a"][iu], ".", ms=2, alpha=0.35)
    m = fid["emp_cross"][iu].max()
    axes[1].plot([0, m], [0, m], "k--", lw=0.8)
    axes[1].set_xlabel("empirical K(i,j)"); axes[1].set_ylabel("paper pipeline K(i,j)")
    axes[1].set_title("cross-kernels, full pipeline")
    axes[2].hist(fid["err_a"], bins=60, alpha=0.75,
                 label=f"(a) full pipeline, median {np.median(fid['err_a']):.3f}")
    axes[2].hist(fid["err_b"], bins=60, alpha=0.75,
                 label=f"(b) oracle correlation, median {np.median(fid['err_b']):.3f}")
    axes[2].set_xlabel("|error| / sqrt(K_ii K_jj)"); axes[2].legend(fontsize=8)
    axes[2].set_title("error attribution")
    fig.suptitle("Experiment B2: cross-kernel error, correlation model vs. Gaussian shape")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_cross.png", dpi=140)
    plt.close(fig)
