"""Run Stage 3 end to end: encode the char-MLP and draw the map.

Usage:  PYTHONPATH=src:. python -m stage3.run_stage3
Writes ledgers, figures, and results.md into stage3/results/.
"""

from __future__ import annotations

import csv
import os
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as sps

from stage2 import data as D
from stage2.calibrate import calibrated_layer
from stage2.model import CharMLP
from stage3.encoder import encode_and_map, static_prune_curve
from stage3.labels import context_properties, label_units, top_contexts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
WEIGHTS = os.path.join(OUT, "char_mlp_weights.npz")

K, D_EMB, HIDDEN, TRAIN_STEPS = 8, 16, 256, 25000
TARGET_DENSITY = 0.35


def get_model(vocab, train_ids, val):
    model = CharMLP(len(vocab), k=K, d=D_EMB, hidden=HIDDEN, seed=0)
    if os.path.exists(WEIGHTS):
        z = np.load(WEIGHTS)
        for p in model._params:
            getattr(model, p)[...] = z[p]
        print(f"loaded cached weights; val loss {model.loss(*val):.4f}")
    else:
        model.fit(train_ids, steps=TRAIN_STEPS, val=val)
        np.savez(WEIGHTS, **{p: getattr(model, p) for p in model._params})
    return model


def write_ledger(path, events):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "action", "removed_ids", "removed_labels",
                    "removed_exkurt", "removed_caps", "parent_id", "rho",
                    "cost", "n_after", "density", "val_loss"])
        for e in events:
            w.writerow([e.step, e.action, "|".join(map(str, e.removed_ids)),
                        "|".join(e.removed_labels),
                        "|".join(f"{x:+.3f}" for x in e.removed_exkurt),
                        "|".join(f"{x:.4f}" for x in e.removed_caps),
                        e.parent_id if e.parent_id is not None else "",
                        f"{e.rho:+.4f}" if e.rho is not None else "",
                        f"{e.cost:.5f}", e.n_after, f"{e.density:.4f}",
                        f"{e.val_loss:.4f}"])


def removed_originals(events, h0):
    out = {}
    for e in events:
        for u in e.removed_ids:
            if u < h0:
                out[u] = e.step
    return out


def loss_at(curve, density):
    ds = np.array([d for d, _ in curve]); ls = np.array([l for _, l in curve])
    idx = np.argmin(np.abs(ds - density))
    return float(ls[idx])


def build_context():
    os.makedirs(OUT, exist_ok=True)
    text = D.load_corpus()
    vocab = D.CharVocab(text)
    ids = vocab.encode(text)
    train_ids, val_ids = D.train_val_split(ids)
    rng = np.random.default_rng(0)
    val = D.sample_contexts(val_ids, K, 8192, rng)
    model = get_model(vocab, train_ids, val)

    calib_ctx, _ = D.sample_contexts(train_ids, K, 8192, np.random.default_rng(10))
    layer = calibrated_layer(model, calib_ctx)
    probe_ctx, probe_tgt = D.sample_contexts(train_ids, K, 32768,
                                             np.random.default_rng(12))

    # ---- unit annotations: labels, kurtosis, top contexts
    h_probe = model.preactivations(probe_ctx)
    z = (h_probe - h_probe.mean(0)) / (h_probe.std(0) + 1e-12)
    exkurt = (z**4).mean(0) - 3.0
    props = context_properties(probe_ctx, probe_tgt, vocab)
    unit_labels, corr, prop_names = label_units(h_probe, props)
    label_counts = Counter(unit_labels)
    print("label counts:", dict(label_counts))
    return dict(model=model, layer=layer, val=val, vocab=vocab,
                probe_ctx=probe_ctx, probe_tgt=probe_tgt, h_probe=h_probe,
                exkurt=exkurt, unit_labels=unit_labels,
                label_counts=label_counts)


def main(part="all"):
    import pickle
    t0 = time.time()
    ctx = build_context()
    model, layer, val = ctx["model"], ctx["layer"], ctx["val"]
    unit_labels, exkurt = ctx["unit_labels"], ctx["exkurt"]
    label_counts, probe_ctx = ctx["label_counts"], ctx["probe_ctx"]
    h_probe, vocab = ctx["h_probe"], ctx["vocab"]

    p_paper = os.path.join(OUT, "map_paper.pkl")
    p_emp = os.path.join(OUT, "map_empirical.pkl")
    if part in ("all", "paper"):
        print("mapping run: paper rho_hat")
        res_paper = encode_and_map(model, layer, unit_labels, exkurt, val,
                                   probe_ctx=probe_ctx, mode="paper",
                                   target_density=TARGET_DENSITY)
        pickle.dump(res_paper, open(p_paper, "wb"))
        write_ledger(os.path.join(OUT, "ledger_paper.csv"), res_paper.events)
    if part in ("all", "empirical"):
        print("mapping run: empirical rho")
        res_emp = encode_and_map(model, layer, unit_labels, exkurt, val,
                                 probe_ctx=probe_ctx[:16384], mode="empirical",
                                 target_density=TARGET_DENSITY)
        pickle.dump(res_emp, open(p_emp, "wb"))
        write_ledger(os.path.join(OUT, "ledger_empirical.csv"), res_emp.events)
    if part not in ("all", "finish"):
        print(f"part {part} done in {time.time() - t0:.0f}s")
        return
    res_paper = pickle.load(open(p_paper, "rb"))
    res_emp = pickle.load(open(p_emp, "rb"))

    # ---- baselines: static prune-only orders
    caps0 = layer.capacities()
    hope_order = list(np.argsort(caps0))
    l1_order = list(np.argsort(np.abs(model.W1).sum(axis=0)))
    rand_order = list(np.random.default_rng(3).permutation(HIDDEN))
    curves = {
        "HOPE full (paper rho)": res_paper.val_curve,
        "HOPE full (empirical rho)": res_emp.val_curve,
        "HOPE prune-only": static_prune_curve(model, hope_order, val, TARGET_DENSITY),
        "L1-norm prune": static_prune_curve(model, l1_order, val, TARGET_DENSITY),
        "random prune": static_prune_curve(model, rand_order, val, TARGET_DENSITY),
    }

    # ---- ledger stability between the two runs
    rem_p = removed_originals(res_paper.events, HIDDEN)
    rem_e = removed_originals(res_emp.events, HIDDEN)
    common = sorted(set(rem_p) & set(rem_e))
    union = set(rem_p) | set(rem_e)
    jaccard = len(common) / max(len(union), 1)
    spear = sps.spearmanr([rem_p[u] for u in common],
                          [rem_e[u] for u in common]).statistic if len(common) > 2 else float("nan")

    # ---- eviction-order statistics (paper run)
    removal_step = rem_p
    by_label = defaultdict(list)
    for u, s in removal_step.items():
        by_label[unit_labels[u]].append(s)
    med_step = {lab: float(np.median(v)) for lab, v in by_label.items()}
    survival = {lab: 1.0 - len(by_label.get(lab, [])) / label_counts[lab]
                for lab in label_counts}
    merges = [e for e in res_paper.events if e.action == "merge"]
    orig_merges = [e for e in merges if all(u < HIDDEN for u in e.removed_ids)]
    same_label = [e for e in orig_merges
                  if e.removed_labels[0] == e.removed_labels[1]]
    removed_k = np.array([abs(k) for u, k in
                          ((u, exkurt[u]) for u in removal_step if u < HIDDEN)])
    surv_mask = np.ones(HIDDEN, bool)
    surv_mask[list(removal_step)] = False
    surviving_k = np.abs(exkurt[surv_mask])

    make_figures(curves, res_paper, res_emp, unit_labels, label_counts,
                 survival, HIDDEN)

    showcase = []
    for lab in ("next=upper", "last=newline", "next=space"):
        us = [u for u in range(HIDDEN) if unit_labels[u] == lab][:1]
        for u in us:
            fate = ("pruned step %d" % removal_step[u]) if u in removal_step \
                else "survives"
            tops = "; ".join(f'"{t}"' for t in
                             top_contexts(h_probe, probe_ctx, vocab, u, k=2))
            showcase.append(f'- unit {u} [{lab}, {fate}]: top contexts {tops}')

    med = np.median
    lines = [
        "# Stage 3 results: the map",
        "",
        f"Model: the Stage 2 char-MLP (val {model.loss(*val):.4f} nats); "
        f"encoded to {TARGET_DENSITY:.0%} density; probe 32,768 contexts.",
        f"Unit labels at |corr| >= 0.25: " +
        ", ".join(f"{k} {v}" for k, v in sorted(label_counts.items())),
        "",
        "## Trajectory (val loss at 50% / 35% density)",
    ] + [
        f"- {name}: {loss_at(c, 0.5):.4f} / {loss_at(c, TARGET_DENSITY):.4f}"
        for name, c in curves.items()
    ] + [
        "",
        "## The eviction order (paper run)",
        f"- actions: {len(res_paper.events)} total, "
        f"{sum(e.action == 'merge' for e in res_paper.events)} merges, "
        f"{sum(e.action == 'prune' for e in res_paper.events)} prunes",
        "- median removal step by label: " +
        ", ".join(f"{k} {v:.0f}" for k, v in
                  sorted(med_step.items(), key=lambda kv: kv[1])),
        "- survival fraction by label: " +
        ", ".join(f"{k} {survival[k]:.2f}" for k, _ in
                  sorted(survival.items(), key=lambda kv: -kv[1])),
        f"- |excess kurtosis|: removed units median {med(removed_k):.3f} vs "
        f"surviving {med(surviving_k):.3f}",
        f"- merges joining two original units: {len(orig_merges)}, of which "
        f"same-label {len(same_label)} "
        f"({len(same_label)/max(len(orig_merges),1):.0%})",
        "",
        "## Robustness to the correlation model",
        f"- removed-set Jaccard (paper vs empirical rho): {jaccard:.3f}",
        f"- removal-order Spearman on common units: {spear:.3f}",
        f"- merges under empirical rho: "
        f"{sum(e.action == 'merge' for e in res_emp.events)} "
        f"(vs {len(merges)} under the MaxEnt warp)",
        "",
        "## Showcase units",
    ] + showcase + [
        "",
        f"Total runtime {time.time() - t0:.0f}s.",
    ]
    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_figures(curves, res_paper, res_emp, unit_labels, label_counts,
                 survival, h0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig5: trajectories
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, c in curves.items():
        d = [x for x, _ in c]; l = [y for _, y in c]
        ax.plot(d, l, lw=1.6 if name.startswith("HOPE full") else 1.1,
                label=name)
    ax.invert_xaxis()
    ax.set_xlabel("model density"); ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Compression trajectories: the ledger as a loss curve")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_trajectory.png", dpi=140)
    plt.close(fig)

    # fig6: the eviction map
    labs = sorted(label_counts)
    cmap = plt.get_cmap("tab10")
    color = {lab: cmap(k % 10) for k, lab in enumerate(labs)}
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), height_ratios=[2, 1])
    ax = axes[0]
    for e in res_paper.events:
        for u, lab, cap in zip(e.removed_ids, e.removed_labels, e.removed_caps):
            if lab == "parent":
                lab_c, mk = "distributed", "s"
            else:
                lab_c, mk = lab, ("o" if e.action == "prune" else "^")
            ax.scatter(e.step, cap, s=22, color=color[lab_c], marker=mk,
                       linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("encoding step"); ax.set_ylabel("capacity at removal (log)")
    ax.set_title("The eviction map: o pruned, ^ merged away, colored by label")
    handles = [plt.Line2D([], [], marker="o", ls="", color=color[l], label=l)
               for l in labs]
    ax.legend(handles=handles, fontsize=7, ncol=3, loc="lower right")

    ax = axes[1]
    counts = {lab: label_counts[lab] for lab in labs}
    alive = dict(counts)
    xs = [1.0]; series = {lab: [1.0] for lab in labs}
    for e in res_paper.events:
        for u, lab in zip(e.removed_ids, e.removed_labels):
            if lab in alive:
                alive[lab] -= 1
        xs.append(e.density)
        for lab in labs:
            series[lab].append(alive[lab] / counts[lab])
    for lab in labs:
        ax.plot(xs, series[lab], color=color[lab], lw=1.4, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel("model density")
    ax.set_ylabel("fraction of label intact")
    ax.set_title("Which knowledge goes first (merges count as consolidation)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_eviction_map.png", dpi=140)
    plt.close(fig)

    # fig7: merge behavior under the two correlation models
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, res, name in ((axes[0], res_paper, "MaxEnt rho_hat"),
                          (axes[1], res_emp, "empirical rho")):
        ms = [e for e in res.events if e.action == "merge"]
        if ms:
            ax.hist([e.rho for e in ms], bins=24)
        ax.set_title(f"{name}: {len(ms)} merges")
        ax.set_xlabel("correlation at merge"); ax.set_ylabel("count")
    fig.suptitle("Merge appetite under the two correlation models")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_merges.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
