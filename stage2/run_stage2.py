from __future__ import annotations
import os
import time
import numpy as np
from stage2 import data as D
from stage2.calibrate import calibrated_layer
from stage2.experiments import (collect_preacts_f32, gaussian_control_preacts,
                                gaussianity_stats, kernel_fidelity, make_figures)
from stage2.model import CharMLP, gradient_check

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

K, D_EMB, HIDDEN = 8, 16, 256
TRAIN_STEPS = 25000
CALIB_N = 8192
EVAL_N = 120000


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    ok, worst = gradient_check()
    assert ok, f"gradient check failed ({worst:.2e})"
    print(f"gradient check passed (worst rel dev {worst:.2e})")

    text = D.load_corpus()
    vocab = D.CharVocab(text)
    ids = vocab.encode(text)
    train_ids, val_ids = D.train_val_split(ids)
    freqs = np.bincount(train_ids, minlength=len(vocab)) / len(train_ids)
    unigram = float(-(freqs[freqs > 0] * np.log(freqs[freqs > 0])).sum())
    print(f"corpus {len(ids):,} chars, vocab {len(vocab)}; "
          f"uniform loss {np.log(len(vocab)):.3f}, unigram entropy {unigram:.3f}")

    rng = np.random.default_rng(0)
    model = CharMLP(len(vocab), k=K, d=D_EMB, hidden=HIDDEN, seed=0)
    val_ctx, val_tgt = D.sample_contexts(val_ids, K, 4096, rng)
    model.fit(train_ids, steps=TRAIN_STEPS, val=(val_ctx, val_tgt))
    print(f"trained in {time.time() - t0:.0f}s")

    calib_ctx, _ = D.sample_contexts(train_ids, K, CALIB_N,
                                     np.random.default_rng(10))
    eval_ctx, _ = D.sample_contexts(train_ids, K, EVAL_N,
                                    np.random.default_rng(11))
    layer = calibrated_layer(model, calib_ctx)

    #experiment A
    h_pre = collect_preacts_f32(model, eval_ctx)
    gauss = gaussianity_stats(h_pre)
    calib_x = model.features(calib_ctx)
    h_ctrl = gaussian_control_preacts(model, calib_x, n=EVAL_N)
    gauss_ctrl = gaussianity_stats(h_ctrl, seed=1)

    mean_err = np.max(np.abs(layer.preact_mean() - gauss["mean"]))
    std_err = np.max(np.abs(layer.preact_std() - gauss["std"]) / gauss["std"])
    print(f"calibration check: max |mean gap| {mean_err:.4f}, "
          f"max rel std gap {std_err:.4f} (calib vs eval slice)")

    #experiment B
    fid = kernel_fidelity(layer, h_pre)
    fid["_exkurt"] = gauss["exkurt"]
    #recompute self kernels from the evaluation slice's own moments
    from minihope.kernels import relu_self_kernel
    oracle_self = relu_self_kernel(gauss["mean"], gauss["std"])
    oracle_rel = np.abs(oracle_self - fid["emp_self"]) / (fid["emp_self"] + 1e-12)

    char_counts, word_counts = D.zipf_tables(text)
    make_figures(OUT, char_counts, word_counts, gauss, gauss_ctrl, fid)

    med = np.median
    lines = [
        "# Stage 2 results: the Gaussian surrogate vs. text",
        "",
        f"Model: char-MLP, context {K}, embed {D_EMB}, hidden {HIDDEN} (ReLU); "
        f"{TRAIN_STEPS} Adam steps on tiny Shakespeare "
        f"({len(ids):,} chars, vocab {len(vocab)}).",
        f"Final val loss {model.loss(val_ctx, val_tgt):.4f} nats/char "
        f"(uniform {np.log(len(vocab)):.3f}, unigram {unigram:.3f}).",
        f"Calibration batch {CALIB_N:,} contexts; evaluation slice "
        f"{EVAL_N:,} contexts.",
        "",
        "## Experiment A: shape of pre-activations",
        f"- excess kurtosis: median {med(gauss['exkurt']):+.3f}, "
        f"90th pct {np.quantile(gauss['exkurt'], 0.9):+.3f}, "
        f"max {gauss['exkurt'].max():+.3f} "
        f"(Gaussian control: median {med(gauss_ctrl['exkurt']):+.3f}, "
        f"max {gauss_ctrl['exkurt'].max():+.3f})",
        f"- |skewness|: median {med(np.abs(gauss['skew'])):.3f}, "
        f"max {np.abs(gauss['skew']).max():.3f} "
        f"(control median {med(np.abs(gauss_ctrl['skew'])):.3f})",
        f"- KS distance to normal: median {med(gauss['ks']):.4f} "
        f"(control median {med(gauss_ctrl['ks']):.4f})",
        "",
        "## Experiment B: kernel fidelity (moments matched by calibration)",
        f"- self-kernel K(i,i) relative error: median "
        f"{med(fid['self_rel']):.4f}, 90th pct "
        f"{np.quantile(fid['self_rel'], 0.9):.4f}, max {fid['self_rel'].max():.4f}",
        f"- self-kernel error with oracle (eval-slice) moments, i.e. pure "
        f"shape error: median {med(oracle_rel):.4f}, 90th pct "
        f"{np.quantile(oracle_rel, 0.9):.4f}, max {oracle_rel.max():.4f}",
        f"- cross-kernel scaled error, (a) full pipeline: median "
        f"{med(fid['err_a']):.4f}, 90th pct {np.quantile(fid['err_a'], 0.9):.4f}",
        f"- cross-kernel scaled error, (b) oracle correlation: median "
        f"{med(fid['err_b']):.4f}, 90th pct {np.quantile(fid['err_b'], 0.9):.4f}",
        f"- correlation model: Spearman(rho_hat, empirical rho) = "
        f"{fid['spearman_rho']:.4f}",
        f"- capacity ordering: Spearman(closed, empirical) = "
        f"{fid['spearman_capacity']:.6f}",
        "",
        f"Total runtime {time.time() - t0:.0f}s.",
    ]
    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
