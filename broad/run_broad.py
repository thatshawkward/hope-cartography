from __future__ import annotations
import os
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as sps

from stage2.corpora import load_masc, load_oanc, sample_contexts_in_spans
from stage2.data import CharVocab
from stage2.calibrate import calibrated_layer
from stage2.experiments import (collect_preacts_f32, gaussian_control_preacts,
                                gaussianity_stats, kernel_fidelity)
from stage2.model import CharMLP
from stage3.encoder import encode_and_map, static_prune_curve
from stage3.labels import context_properties, label_units, top_contexts
from stage3.run_stage3 import loss_at, removed_originals, write_ledger

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

K, D_EMB, HIDDEN = 12, 24, 384
STEPS, BATCH, LR = 60000, 256, 1.5e-3
TARGET_DENSITY = 0.35

COARSE = {"conversation": {"face-to-face", "telephone"},
          "formal-spoken": {"court-transcript", "debate-transcript"}}
N_REGISTERS = 10

#Shakespeare pilot numbers (stage2/results, stage3/results) for the side by side replication table.
PILOT = dict(kurt_med=-0.008, kurt_p90=0.447, self_err=0.076,
             cross_a=0.094, cross_b=0.123, cap_spear=0.9822,
             merges_paper=0, merges_emp=17, jaccard=0.898, order_spear=0.974)


def get_corpus(which=None):
    name = "oanc" if (which == "oanc" or
                      (which is None and os.path.isdir("data/oanc"))) else "masc"
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, f"corpus_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        from stage2.corpora import Corpus
        text = z["text"].tobytes().decode("utf-8")
        docs = [(str(g), int(a), int(b)) for g, a, b in
                zip(z["genres"], z["starts"], z["ends"])]
        c = Corpus(name=name, text=text, docs=docs)
    else:
        c = load_oanc("data/oanc") if name == "oanc" else load_masc()
        np.savez_compressed(
            cache, text=np.frombuffer(c.text.encode("utf-8"), dtype=np.uint8),
            genres=np.array([d[0] for d in c.docs]),
            starts=np.array([d[1] for d in c.docs]),
            ends=np.array([d[2] for d in c.docs]))
    c.split_spans(val_fraction=0.1, seed=0)
    return c


def context(which=None):
    os.makedirs(OUT, exist_ok=True)
    corpus = get_corpus(which)
    vocab = CharVocab(corpus.text)
    ids = vocab.encode(corpus.text)
    rng = np.random.default_rng(0)
    val = sample_contexts_in_spans(ids, corpus.val_docs, K, 8192, rng)
    model = CharMLP(len(vocab), k=K, d=D_EMB, hidden=HIDDEN, seed=0)
    wpath = os.path.join(OUT, f"weights_{corpus.name}.npz")
    if os.path.exists(wpath):
        z = np.load(wpath)
        for p in model._params:
            getattr(model, p)[...] = z[p]
    return dict(corpus=corpus, vocab=vocab, ids=ids, val=val, model=model,
                wpath=wpath)


def part_train(ctx):
    corpus, ids, model, val = ctx["corpus"], ctx["ids"], ctx["model"], ctx["val"]
    freqs = np.bincount(ids, minlength=len(ctx["vocab"])) / len(ids)
    uni = float(-(freqs[freqs > 0] * np.log(freqs[freqs > 0])).sum())
    print(f"{corpus.name}: {len(corpus.text):,} chars, {len(corpus.docs)} docs, "
          f"vocab {len(ctx['vocab'])}; uniform {np.log(len(ctx['vocab'])):.3f}, "
          f"unigram {uni:.3f}")
    rng = np.random.default_rng(1)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bctx, btgt = sample_contexts_in_spans(ids, corpus.train_docs, K, BATCH, rng)
        _, grads = model.loss_and_grads(bctx, btgt)
        model.adam_step(grads, lr=LR)
        if step % 4000 == 0 or step == 1:
            print(f"step {step:>6}  val {model.loss(*val):.4f}", flush=True)
    np.savez(ctx["wpath"], **{p: getattr(model, p) for p in model._params})
    print(f"trained in {time.time() - t0:.0f}s; val {model.loss(*val):.4f}")


def _train_sample(ctx, n, seed):
    return sample_contexts_in_spans(ctx["ids"], ctx["corpus"].train_docs, K, n,
                                    np.random.default_rng(seed))


def part_surrogate(ctx):
    model = ctx["model"]
    calib_ctx, _ = _train_sample(ctx, 8192, 10)
    eval_ctx, _ = _train_sample(ctx, 100000, 11)
    layer = calibrated_layer(model, calib_ctx)
    h = collect_preacts_f32(model, eval_ctx)
    g = gaussianity_stats(h)
    ctrl = gaussianity_stats(gaussian_control_preacts(
        model, model.features(calib_ctx), n=100000), seed=1)
    fid = kernel_fidelity(layer, h)
    med = np.median
    metrics = dict(
        kurt_med=float(med(g["exkurt"])), kurt_p90=float(np.quantile(g["exkurt"], .9)),
        kurt_max=float(g["exkurt"].max()), kurt_ctrl_med=float(med(ctrl["exkurt"])),
        skew_med=float(med(np.abs(g["skew"]))), ks_med=float(med(g["ks"])),
        ks_ctrl_med=float(med(ctrl["ks"])),
        self_err=float(med(fid["self_rel"])), self_p90=float(np.quantile(fid["self_rel"], .9)),
        cross_a=float(med(fid["err_a"])), cross_b=float(med(fid["err_b"])),
        cap_spear=float(fid["spearman_capacity"]), rho_spear=float(fid["spearman_rho"]))
    pickle.dump(metrics, open(os.path.join(OUT, "surrogate.pkl"), "wb"))
    print("surrogate replication:", {k: round(v, 4) for k, v in metrics.items()})


def part_map(ctx):
    model, val = ctx["model"], ctx["val"]
    calib_ctx, _ = _train_sample(ctx, 8192, 10)
    layer = calibrated_layer(model, calib_ctx)
    probe_ctx, probe_tgt = _train_sample(ctx, 32768, 12)
    h_probe = model.preactivations(probe_ctx)
    z = (h_probe - h_probe.mean(0)) / (h_probe.std(0) + 1e-12)
    exkurt = (z**4).mean(0) - 3.0
    labels, _, _ = label_units(h_probe, context_properties(probe_ctx, probe_tgt,
                                                           ctx["vocab"]))
    print("label counts:", dict(Counter(labels)))
    pickle.dump(dict(labels=labels, exkurt=exkurt),
                open(os.path.join(OUT, "units.pkl"), "wb"))
    print("mapping run: paper rho_hat")
    rp = encode_and_map(model, layer, labels, exkurt, val, probe_ctx=probe_ctx,
                        mode="paper", target_density=TARGET_DENSITY,
                        verbose_every=60)
    pickle.dump(rp, open(os.path.join(OUT, "map_paper.pkl"), "wb"))
    write_ledger(os.path.join(OUT, "ledger_paper.csv"), rp.events)
    print("mapping run: empirical rho")
    re_ = encode_and_map(model, layer, labels, exkurt, val,
                         probe_ctx=probe_ctx[:16384], mode="empirical",
                         target_density=TARGET_DENSITY, verbose_every=60)
    pickle.dump(re_, open(os.path.join(OUT, "map_empirical.pkl"), "wb"))
    write_ledger(os.path.join(OUT, "ledger_empirical.csv"), re_.events)


def _registers(corpus, min_chars=60000):
    vols = corpus.genre_volumes(corpus.train_docs)
    consumed = set().union(*COARSE.values())
    regs = {}
    for name, genres in COARSE.items():
        if sum(vols.get(g, 0) for g in genres) >= min_chars:
            regs[name] = set(genres)
    for g, v in vols.items():
        if g not in consumed and v >= min_chars and len(regs) < N_REGISTERS:
            regs[g] = {g}
    return regs


def part_finish(ctx, t0):
    corpus, vocab, model, val = (ctx["corpus"], ctx["vocab"], ctx["model"],
                                 ctx["val"])
    metrics = pickle.load(open(os.path.join(OUT, "surrogate.pkl"), "rb"))
    units = pickle.load(open(os.path.join(OUT, "units.pkl"), "rb"))
    labels, exkurt = units["labels"], units["exkurt"]
    rp = pickle.load(open(os.path.join(OUT, "map_paper.pkl"), "rb"))
    re_ = pickle.load(open(os.path.join(OUT, "map_empirical.pkl"), "rb"))
    calib_ctx, _ = _train_sample(ctx, 8192, 10)
    layer = calibrated_layer(model, calib_ctx)
    probe_ctx, probe_tgt = _train_sample(ctx, 32768, 12)
    h_probe = model.preactivations(probe_ctx)

    #baselines
    caps0 = layer.capacities()
    curves = {
        "HOPE full (paper rho)": rp.val_curve,
        "HOPE full (empirical rho)": re_.val_curve,
        "HOPE prune-only": static_prune_curve(model, list(np.argsort(caps0)),
                                              val, TARGET_DENSITY),
        "L1-norm prune": static_prune_curve(
            model, list(np.argsort(np.abs(model.W1).sum(0))), val, TARGET_DENSITY),
        "random prune": static_prune_curve(
            model, list(np.random.default_rng(3).permutation(HIDDEN)), val,
            TARGET_DENSITY),
    }

    #map robustness and eviction order
    rem_p = removed_originals(rp.events, HIDDEN)
    rem_e = removed_originals(re_.events, HIDDEN)
    common = sorted(set(rem_p) & set(rem_e))
    jac = len(common) / max(len(set(rem_p) | set(rem_e)), 1)
    spear = sps.spearmanr([rem_p[u] for u in common],
                          [rem_e[u] for u in common]).statistic
    label_counts = Counter(labels)
    by_label = defaultdict(list)
    for u, s in rem_p.items():
        by_label[labels[u]].append(s)
    survival = {lab: 1.0 - len(by_label.get(lab, [])) / label_counts[lab]
                for lab in label_counts}
    merges_p = sum(e.action == "merge" for e in rp.events)
    merges_e = sum(e.action == "merge" for e in re_.events)
    surv_mask = np.ones(HIDDEN, bool); surv_mask[list(rem_p)] = False
    kurt_removed = float(np.median([abs(exkurt[u]) for u in rem_p if u < HIDDEN]))
    kurt_surv = float(np.median(np.abs(exkurt[surv_mask])))

    #the register conditional map
    regs = _registers(corpus)
    reg_caps, reg_names = [], []
    for name, genres in regs.items():
        spans = [d for d in corpus.train_docs if d[0] in genres]
        rctx, _ = sample_contexts_in_spans(ctx["ids"], spans, K, 6000,
                                           np.random.default_rng(20 + len(reg_names)))
        reg_caps.append(calibrated_layer(model, rctx).capacities())
        reg_names.append(name)
    reg_caps = np.stack(reg_caps)                       # (R, H)
    R = len(reg_names)
    spear_mx = np.ones((R, R)); jac_mx = np.ones((R, R))
    cores = [set(np.argsort(c)[-HIDDEN // 2:]) for c in reg_caps]
    for a in range(R):
        for b in range(a + 1, R):
            s = sps.spearmanr(reg_caps[a], reg_caps[b]).statistic
            j = len(cores[a] & cores[b]) / len(cores[a] | cores[b])
            spear_mx[a, b] = spear_mx[b, a] = s
            jac_mx[a, b] = jac_mx[b, a] = j
    variability = reg_caps.max(0) / np.maximum(reg_caps.min(0), 1e-9)
    var_units = np.argsort(variability)[::-1][:5]
    var_lines = []
    for u in var_units:
        hi = reg_names[int(np.argmax(reg_caps[:, u]))]
        lo = reg_names[int(np.argmin(reg_caps[:, u]))]
        tc = top_contexts(h_probe, probe_ctx, vocab, int(u), k=1)[0]
        var_lines.append(f'- unit {u} [{labels[u]}] x{variability[u]:.1f}: '
                         f'core in {hi}, slack in {lo}; top context "{tc}"')

    make_figures(curves, rp, labels, label_counts, spear_mx, jac_mx, reg_names)

    med = np.median
    off = spear_mx[np.triu_indices(R, 1)]
    joff = jac_mx[np.triu_indices(R, 1)]
    lines = [
        f"# Broad-corpus run: {corpus.name.upper()}",
        "",
        f"Corpus: {len(corpus.text):,} chars, {len(corpus.docs)} docs, "
        f"{len(corpus.genre_volumes())} genres; model k={K}, d={D_EMB}, "
        f"hidden={HIDDEN}; val loss {model.loss(*val):.4f} nats/char.",
        "",
        "## Surrogate replication (this corpus vs. Shakespeare pilot)",
        f"- excess kurtosis med/p90: {metrics['kurt_med']:+.3f}/"
        f"{metrics['kurt_p90']:+.3f} (pilot {PILOT['kurt_med']:+.3f}/"
        f"{PILOT['kurt_p90']:+.3f}); max {metrics['kurt_max']:+.1f}; "
        f"control med {metrics['kurt_ctrl_med']:+.3f}",
        f"- self-kernel err med/p90: {metrics['self_err']:.3f}/"
        f"{metrics['self_p90']:.3f} (pilot {PILOT['self_err']:.3f})",
        f"- cross-kernel err, full pipeline / oracle rho: {metrics['cross_a']:.3f}"
        f" / {metrics['cross_b']:.3f} (pilot {PILOT['cross_a']:.3f} / "
        f"{PILOT['cross_b']:.3f})",
        f"- capacity ordering Spearman: {metrics['cap_spear']:.4f} "
        f"(pilot {PILOT['cap_spear']:.4f}); correlation-model Spearman "
        f"{metrics['rho_spear']:.3f}",
        "",
        "## Trajectory (val loss at 50% / 35% density)",
    ] + [f"- {n}: {loss_at(c, 0.5):.4f} / {loss_at(c, TARGET_DENSITY):.4f}"
         for n, c in curves.items()] + [
        "",
        "## The map",
        f"- merges: {merges_p} under the MaxEnt warp, {merges_e} under "
        f"empirical rho (pilot: {PILOT['merges_paper']} / {PILOT['merges_emp']})",
        f"- robustness: removed-set Jaccard {jac:.3f}, order Spearman "
        f"{spear:.3f} (pilot {PILOT['jaccard']:.3f} / {PILOT['order_spear']:.3f})",
        "- survival by label: " + ", ".join(
            f"{k} {survival[k]:.2f}" for k, _ in
            sorted(survival.items(), key=lambda kv: -kv[1])),
        f"- |excess kurtosis|: removed med {kurt_removed:.3f} vs surviving "
        f"{kurt_surv:.3f}",
        "",
        f"## The register-conditional map ({R} registers)",
        f"- capacity Spearman across registers: median {med(off):.4f}, "
        f"min {off.min():.4f}",
        f"- top-half core overlap (Jaccard): median {med(joff):.3f}, "
        f"min {joff.min():.3f}",
        "- most register-variable units (max/min capacity ratio):",
    ] + var_lines + ["", f"Total runtime {time.time() - t0:.0f}s."]
    with open(os.path.join(OUT, "results_broad.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_figures(curves, rp, labels, label_counts, spear_mx, jac_mx, reg_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, c in curves.items():
        ax.plot([x for x, _ in c], [y for _, y in c],
                lw=1.6 if name.startswith("HOPE full") else 1.1, label=name)
    ax.invert_xaxis(); ax.legend(fontsize=8)
    ax.set_xlabel("model density"); ax.set_ylabel("val loss (nats/char)")
    ax.set_title("Broad-corpus compression trajectories")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_b2_trajectory.png", dpi=140)
    plt.close(fig)

    labs = sorted(label_counts)
    cmap = plt.get_cmap("tab10")
    color = {lab: cmap(k % 10) for k, lab in enumerate(labs)}
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), height_ratios=[2, 1])
    ax = axes[0]
    for e in rp.events:
        for u, lab, cap in zip(e.removed_ids, e.removed_labels, e.removed_caps):
            lab_c = "distributed" if lab == "parent" else lab
            mk = "o" if e.action == "prune" else "^"
            ax.scatter(e.step, cap, s=18, color=color[lab_c], marker=mk,
                       linewidths=0)
    ax.set_yscale("log"); ax.set_xlabel("encoding step")
    ax.set_ylabel("capacity at removal (log)")
    ax.set_title("The eviction map (OANC run): o pruned, ^ merged away")
    handles = [plt.Line2D([], [], marker="o", ls="", color=color[l], label=l)
               for l in labs]
    ax.legend(handles=handles, fontsize=7, ncol=3, loc="lower right")
    ax = axes[1]
    alive = dict(label_counts); xs = [1.0]
    series = {lab: [1.0] for lab in labs}
    for e in rp.events:
        for u, lab in zip(e.removed_ids, e.removed_labels):
            if lab in alive:
                alive[lab] -= 1
        xs.append(e.density)
        for lab in labs:
            series[lab].append(alive[lab] / label_counts[lab])
    for lab in labs:
        ax.plot(xs, series[lab], color=color[lab], lw=1.3, label=lab)
    ax.invert_xaxis(); ax.set_xlabel("model density")
    ax.set_ylabel("fraction of label intact")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_b3_eviction_map.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, mx, title, vmin in ((axes[0], spear_mx, "capacity Spearman", None),
                                (axes[1], jac_mx, "top-half core Jaccard", None)):
        im = ax.imshow(mx, vmin=vmin, cmap="viridis")
        ax.set_xticks(range(len(reg_names)))
        ax.set_xticklabels(reg_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(reg_names)))
        ax.set_yticklabels(reg_names, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("The register-conditional map: same weights, different calibration")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_b4_registers.png", dpi=140)
    plt.close(fig)


def main(part="all", which=None):
    t0 = time.time()
    ctx = context(which)
    if part in ("all", "train") and not os.path.exists(ctx["wpath"]):
        part_train(ctx)
        z = np.load(ctx["wpath"])
        for p in ctx["model"]._params:
            getattr(ctx["model"], p)[...] = z[p]
    elif part == "train":
        print("weights already cached:", ctx["wpath"])
    if part in ("all", "surrogate"):
        part_surrogate(ctx)
    if part in ("all", "map"):
        part_map(ctx)
    if part in ("all", "finish"):
        part_finish(ctx, t0)
    print(f"[{part}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all",
         sys.argv[2] if len(sys.argv) > 2 else None)
