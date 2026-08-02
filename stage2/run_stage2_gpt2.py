"""Stage 2 Experiment A with a pretrained GPT-2 (124M) as the model"""

from __future__ import annotations
import argparse
import os
import time
import numpy as np
from stage2 import data as D
from stage2.streaming import StreamingShapeStats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

MODEL_NAME = "gpt2"
WINDOW = 1024
EVAL_WINDOWS = 118     
CALIB_WINDOWS = 8    
KS_SUB = 20000
BATCH = 2
CTRL_CHUNK = 16384



def _forward_batches(model, windows, device, hooks, want_loss=False, tag=""):
    import torch
    handles = [m.register_forward_hook(fn) for m, fn in hooks]
    loss_sum = ntok = 0
    t0 = time.time()
    n = windows.shape[0]
    try:
        with torch.no_grad():
            for i, s in enumerate(range(0, n, BATCH)):
                x = torch.from_numpy(windows[s:s + BATCH]).to(device)
                if want_loss:
                    out = model(x, labels=x)
                    k = x.shape[0] * (x.shape[1] - 1)
                    loss_sum += float(out.loss) * k
                    ntok += k
                else:
                    model(x)
                done = min(s + BATCH, n)
                if tag and (done == n or i % 8 == 7):
                    print(f"  {tag}: {done}/{n} windows ({time.time() - t0:.0f}s)",
                          flush=True)
    finally:
        for h in handles:
            h.remove()
    return loss_sum / ntok if ntok else None


def _to_2d(t):
    import torch
    return t.detach().to("cpu", torch.float32).numpy().reshape(-1, t.shape[-1])


def out_streamer(stats):
    return lambda mod, inp, out: stats.update(_to_2d(out))


def in_collector(store):
    return lambda mod, inp, out: store.append(_to_2d(inp[0]))


def col_collector(store, unit):
    return lambda mod, inp, out: store.append(_to_2d(out)[:, unit].copy())


def _check_capture_semantics(model, fc, window, device):
    import torch
    grabbed = {}

    def hook(mod, inp, out):
        grabbed["x"], grabbed["h"] = _to_2d(inp[0]), _to_2d(out)

    handle = fc.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(torch.from_numpy(window[None]).to(device))
    finally:
        handle.remove()
    W = fc.weight.detach().cpu().numpy()
    b = fc.bias.detach().cpu().numpy()
    return float(np.abs(grabbed["x"] @ W + b - grabbed["h"]).max())




def control_stats(calib_x, W, b, n, ks_sub, draw_seed):
    """Stream stats of xg @ W + b for iid xg ~ N(mean, cov) of calib_x """
    mu = calib_x.mean(axis=0)
    cov = np.cov(calib_x, rowvar=False)
    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        jitter = 1e-6 * float(np.trace(cov)) / cov.shape[0]
        L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
    mu32, L32 = mu.astype(np.float32), L.astype(np.float32)
    rng = np.random.default_rng(draw_seed)
    cs = StreamingShapeStats(n, W.shape[1], ks_subsample=ks_sub, seed=1)
    for s in range(0, n, CTRL_CHUNK):
        m = min(CTRL_CHUNK, n - s)
        xg = mu32 + rng.standard_normal((m, mu.size), dtype=np.float32) @ L32.T
        cs.update(xg @ W + b)
    return cs


def make_figures(outdir, real, ctrl, pick_lu, names, qq_z, exkurt_of_pick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    cat = lambda key, rs: np.concatenate([r[key] for r in rs])
    ex, exc = cat("exkurt", real), cat("exkurt", ctrl)
    sk, skc = cat("skew", real), cat("skew", ctrl)
    ks, ksc = cat("ks", real), cat("ks", ctrl)

    #fig 2 analog: pooled histograms + QQ of extreme/median/most Gaussian units
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    hi = np.quantile(ex, 0.99)
    bins = np.linspace(min(ex.min(), -1), hi, 50)
    axes[0, 0].hist(np.clip(ex, None, hi), bins=bins, alpha=0.75, label="real text")
    axes[0, 0].hist(np.clip(exc, None, hi), bins=bins, alpha=0.75,
                    label="Gaussian control")
    axes[0, 0].set_xlabel(f"excess kurtosis (clipped at p99; max {ex.max():+.0f})")
    axes[0, 0].legend()
    lo_s, hi_s = np.quantile(sk, [0.005, 0.995])
    bins = np.linspace(lo_s, hi_s, 50)
    axes[0, 1].hist(np.clip(sk, lo_s, hi_s), bins=bins, alpha=0.75, label="real text")
    axes[0, 1].hist(np.clip(skc, lo_s, hi_s), bins=bins, alpha=0.75, label="control")
    axes[0, 1].set_xlabel(f"skewness (clipped at p0.5/p99.5; range "
                          f"{sk.min():+.1f}..{sk.max():+.1f})")
    axes[0, 1].legend()
    axes[0, 2].hist(ks, bins=50, alpha=0.75, label="real text")
    axes[0, 2].hist(ksc, bins=50, alpha=0.75, label="control")
    axes[0, 2].set_xlabel("KS distance to normal")
    axes[0, 2].legend()
    qs = np.linspace(0.001, 0.999, 400)
    tq = sps.norm.ppf(qs)
    for ax, (l, u), name, z, ek in zip(axes[1], pick_lu, names, qq_z,
                                       exkurt_of_pick):
        eq = np.quantile(z, qs)
        ax.plot(tq, eq, lw=1.2)
        ax.plot(tq, tq, "k--", lw=0.8)
        ax.set_title(f"{name}: block {l + 1}, unit {u} (exkurt {ek:+.2f})",
                     fontsize=9)
        ax.set_xlabel("normal quantiles")
        ax.set_ylabel("empirical quantiles")
    fig.suptitle("Experiment A on GPT-2: pre-activation shape on real text "
                 "vs. the Gaussian fiction")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_gaussianity_gpt2.png", dpi=140)
    plt.close(fig)

    # depth profile: how far from Gaussian, block by block
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    xs = np.arange(1, len(real) + 1)
    panels = ((axes[0], "exkurt", False, "excess kurtosis"),
              (axes[1], "skew", True, "|skewness|"),
              (axes[2], "ks", False, "KS distance to normal"))
    for ax, key, absv, label in panels:
        for rs, nm in ((real, "real text"), (ctrl, "Gaussian control")):
            vals = [np.abs(r[key]) if absv else r[key] for r in rs]
            med = [np.median(v) for v in vals]
            ax.plot(xs, med, "-o", ms=3, label=nm)
            ax.fill_between(xs, [np.quantile(v, 0.1) for v in vals],
                            [np.quantile(v, 0.9) for v in vals], alpha=0.2)
        ax.set_xlabel("block")
        ax.set_ylabel(label)
        ax.set_xticks(xs)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment A on GPT-2 by depth "
                 "(median across units; band 10th-90th pct)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2b_gpt2_depth.png", dpi=140)
    plt.close(fig)



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (6 eval / 2 calib windows, KS on 4000 rows)")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args(argv)
    eval_w, calib_w, ks_sub = (6, 2, 4000) if args.smoke else \
        (EVAL_WINDOWS, CALIB_WINDOWS, KS_SUB)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    import torch
    import transformers
    transformers.logging.set_verbosity_error()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model = model.to(device).eval()
    blocks = model.transformer.h
    fcs = [blk.mlp.c_fc for blk in blocks]
    n_layer = len(blocks)
    d_ff = fcs[0].weight.shape[1]

    text = D.load_corpus()
    ids = np.asarray(tok(text)["input_ids"], dtype=np.int64)
    need = (eval_w + calib_w) * WINDOW
    assert ids.size >= need, f"corpus too small: {ids.size} tokens < {need}"
    eval_win = ids[:eval_w * WINDOW].reshape(eval_w, WINDOW)
    calib_win = ids[eval_w * WINDOW:need].reshape(calib_w, WINDOW)
    n_eval = eval_w * WINDOW
    print(f"corpus {len(text):,} chars -> {ids.size:,} BPE tokens "
          f"(vocab {len(tok)}); eval {n_eval:,} positions, "
          f"calib {calib_w * WINDOW:,}, window {WINDOW}, device {device.type}")

    dev = _check_capture_semantics(model, fcs[0], eval_win[0], device)
    print(f"capture check passed: c_fc out == in @ W + b (max dev {dev:.2e})")

    calib_stores = [[] for _ in range(n_layer)]
    _forward_batches(model, calib_win, device,
                     [(fc, in_collector(st)) for fc, st in zip(fcs, calib_stores)],
                     tag="calib")
    calib_x = [np.concatenate(st) for st in calib_stores]

    stats = [StreamingShapeStats(n_eval, d_ff, ks_subsample=ks_sub, seed=0)
             for _ in range(n_layer)]
    loss = _forward_batches(model, eval_win, device,
                            [(fc, out_streamer(s)) for fc, s in zip(fcs, stats)],
                            want_loss=True, tag="eval")
    print(f"GPT-2 sanity: {loss:.4f} nats/token on the eval slice "
          f"(ppl {np.exp(loss):.1f}); collected in {time.time() - t0:.0f}s")

    real = []
    for s in stats:
        r = s.finalize()
        r.pop("zsub")
        s.sub = None
        real.append(r)

    ctrl = []
    for l, fc in enumerate(fcs):
        W = fc.weight.detach().cpu().numpy()
        b = fc.bias.detach().cpu().numpy()
        cs = control_stats(calib_x[l], W, b, n_eval, ks_sub, draw_seed=1000 + l)
        c = cs.finalize()
        c.pop("zsub")
        cs.sub = None
        ctrl.append(c)
        print(f"  control: block {l + 1}/{n_layer} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    ex_all = np.concatenate([r["exkurt"] for r in real])
    order = np.argsort(np.abs(ex_all))
    picks = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    pick_lu = [divmod(p, d_ff) for p in picks]
    names = ["most non-Gaussian unit", "median unit", "most Gaussian unit"]
    qq_stores = [[] for _ in picks]
    _forward_batches(model, eval_win, device,
                     [(fcs[l], col_collector(st, u))
                      for st, (l, u) in zip(qq_stores, pick_lu)], tag="qq")
    qq_z = [(np.concatenate(st) - real[l]["mean"][u]) / real[l]["std"][u]
            for st, (l, u) in zip(qq_stores, pick_lu)]

    make_figures(args.outdir, real, ctrl, pick_lu, names, qq_z,
                 [ex_all[p] for p in picks])

    med, q = np.median, np.quantile
    ex_c = np.concatenate([c["exkurt"] for c in ctrl])
    sk_all = np.concatenate([r["skew"] for r in real])
    sk_c = np.concatenate([c["skew"] for c in ctrl])
    ks_all = np.concatenate([r["ks"] for r in real])
    ks_c = np.concatenate([c["ks"] for c in ctrl])
    lines = [
        "# Stage 2, Experiment A rerun: GPT-2 on tiny Shakespeare",
        "",
        f"Model: pretrained GPT-2 small ({MODEL_NAME}: {n_layer} blocks, "
        f"d_model {model.config.n_embd}, d_ff {d_ff}, GELU). Pre-activations "
        f"are each block's mlp.c_fc output -- the input to the GELU, the "
        f"analog of the char-MLP's x @ W1 + b1.",
        f"Corpus: tiny Shakespeare, {len(text):,} chars -> {ids.size:,} GPT-2 "
        f"BPE tokens (vocab {len(tok)}).",
        f"Sanity: {loss:.4f} nats/token (ppl {np.exp(loss):.1f}) on the eval "
        f"slice.",
        f"Eval slice {n_eval:,} positions ({eval_w} windows x {WINDOW}); "
        f"calibration slice {calib_w * WINDOW:,} positions, disjoint.",
        "Control per block: iid Gaussian inputs with the calibration slice's "
        "MLP-input mean/covariance through the same c_fc weights "
        "(pre-activations exactly Gaussian).",
        "",
        f"## Experiment A, pooled over all {n_layer} x {d_ff} units",
        f"- excess kurtosis: median {med(ex_all):+.3f}, "
        f"90th pct {q(ex_all, 0.9):+.3f}, max {ex_all.max():+.1f} "
        f"(Gaussian control: median {med(ex_c):+.3f}, max {ex_c.max():+.3f})",
        f"- |skewness|: median {med(np.abs(sk_all)):.3f}, "
        f"max {np.abs(sk_all).max():.1f} "
        f"(control median {med(np.abs(sk_c)):.3f})",
        f"- KS distance to normal: median {med(ks_all):.4f} "
        f"(control median {med(ks_c):.4f})",
        "",
        "## By block",
        "| block | exkurt med | exkurt p90 | exkurt max | skew med (abs) "
        "| KS med | KS med (ctrl) |",
        "|---|---|---|---|---|---|---|",
    ]
    for l, (r, c) in enumerate(zip(real, ctrl)):
        lines.append(
            f"| {l + 1} | {med(r['exkurt']):+.3f} | {q(r['exkurt'], 0.9):+.3f} "
            f"| {r['exkurt'].max():+.1f} | {med(np.abs(r['skew'])):.3f} "
            f"| {med(r['ks']):.4f} | {med(c['ks']):.4f} |")
    lines += [
        "",
        f"Total runtime {time.time() - t0:.0f}s (torch {torch.__version__}, "
        f"transformers {transformers.__version__}, device {device.type}).",
    ]
    with open(os.path.join(args.outdir, "results_gpt2.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
