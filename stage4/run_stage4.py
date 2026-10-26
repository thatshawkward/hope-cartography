"""Run Stage 4: trajectory fine-tuning over the stage-3 map.

Usage:  PYTHONPATH=src:. python -m stage4.run_stage4 [base|pilot|arms|finish|all]
Writes curves, ledgers, figures, and results.md into stage4/results/.

Arms, at equal recovery budget per mode:
  base      stage-3 greedy encode, no recovery (also records merge parents)
  A         base plan replayed verbatim, recovery blocks at density checkpoints
  B         greedy re-planned: recovery then recalibration at each checkpoint,
            so later evictions are priced on the adapted network
  end       base plan to target density, one recovery block of the full budget
  capacity/l1/random  static prune orders given the same checkpoint recovery
"""

from __future__ import annotations

import csv
import os
import pickle
import time

import numpy as np
from scipy import stats as sps

from stage2 import data as D
from stage2.calibrate import calibrated_layer
from stage3.encoder import encode_and_map
from stage3.run_stage3 import (K, TARGET_DENSITY, build_context, loss_at,
                               removed_originals, write_ledger)
from stage3.writeback import clone_model
from stage4.finetune import FTLog, checkpoint_densities, ft_block
from stage4.replay import replay, replay_prune_order

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

CHECKPOINTS = checkpoint_densities(0.95, TARGET_DENSITY, 0.05)
FT_STEPS = 750          # per recovery block; chosen by the pilot sweep
FT_LR = 3e-4
MODES = ("paper", "empirical")
H0 = 256


def base_path(mode):
    return os.path.join(OUT, f"base_{mode}.pkl")


def load_base(mode):
    with open(base_path(mode), "rb") as fh:
        return pickle.load(fh)


def fresh(ctx):
    return clone_model(ctx["model"])


def block_seed(idx):
    # same seed per block index in every arm: paired recovery noise
    return 1000 + idx


def run_base(ctx):
    for mode in MODES:
        probe = ctx["probe_ctx"] if mode == "paper" else ctx["probe_ctx"][:16384]
        print(f"base mapping run: {mode}")
        res = encode_and_map(fresh(ctx), ctx["layer"], ctx["unit_labels"],
                             ctx["exkurt"], ctx["val"], probe_ctx=probe,
                             mode=mode, target_density=TARGET_DENSITY)
        with open(base_path(mode), "wb") as fh:
            pickle.dump(res, fh)
        write_ledger(os.path.join(OUT, f"ledger_base_{mode}.csv"), res.events)


def run_pilot(ctx):
    """recovery response at one mid-trajectory point, to pick FT_STEPS/FT_LR"""
    res = load_base("empirical")
    upto = [e for e in res.events if e.density >= 0.60]
    model, curve = replay(fresh(ctx), upto, res.parents, H0,
                          ctx["layer"].eps, ctx["val"])
    v0 = curve[-1][1]
    print(f"pilot at density {curve[-1][0]:.3f}, val {v0:.4f} "
          f"(uncompressed {curve[0][1]:.4f})")
    print(f"{'steps':>6} {'lr':>8} {'val':>8} {'recovered':>10}")
    for lr in (3e-4, 1.5e-3):
        for steps in (250, 500, 1000, 2000):
            m = clone_model(model)
            ft_block(m, ctx["train_ids"], steps=steps, lr=lr, seed=999)
            v = m.loss(*ctx["val"])
            print(f"{steps:>6} {lr:>8.1e} {v:>8.4f} {v0 - v:>10.4f}")


def arm_a(ctx, mode, res, log):
    state = dict(idx=0)

    def hook(model, density, step):
        v0 = model.loss(*ctx["val"])
        ft_block(model, ctx["train_ids"], steps=FT_STEPS, lr=FT_LR,
                 seed=block_seed(state["idx"]))
        log.add("A", mode, density, FT_STEPS, FT_LR, v0, model.loss(*ctx["val"]))
        state["idx"] += 1
        return model

    _, curve = replay(fresh(ctx), res.events, res.parents, H0,
                      ctx["layer"].eps, ctx["val"],
                      checkpoints=CHECKPOINTS, hook=hook)
    return curve


def arm_b(ctx, mode, log, drift):
    calib_ctx, _ = D.sample_contexts(ctx["train_ids"], K, 8192,
                                     np.random.default_rng(10))
    state = dict(idx=0)

    def hook(model, layer, density, step):
        v0 = model.loss(*ctx["val"])
        caps_pre = layer.capacities()
        ft_block(model, ctx["train_ids"], steps=FT_STEPS, lr=FT_LR,
                 seed=block_seed(state["idx"]))
        new_layer = calibrated_layer(model, calib_ctx)
        caps_post = new_layer.capacities()
        log.add("B", mode, density, FT_STEPS, FT_LR, v0, model.loss(*ctx["val"]))
        drift.append(dict(mode=mode, density=density,
                          spearman=float(sps.spearmanr(caps_pre,
                                                       caps_post).statistic)))
        state["idx"] += 1
        return model, new_layer

    probe = ctx["probe_ctx"] if mode == "paper" else ctx["probe_ctx"][:16384]
    res = encode_and_map(fresh(ctx), ctx["layer"], ctx["unit_labels"],
                         ctx["exkurt"], ctx["val"], probe_ctx=probe, mode=mode,
                         target_density=TARGET_DENSITY,
                         checkpoints=CHECKPOINTS, checkpoint_hook=hook)
    with open(os.path.join(OUT, f"armB_{mode}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    write_ledger(os.path.join(OUT, f"ledger_armB_{mode}.csv"), res.events)
    return res.val_curve


def arm_end(ctx, mode, res, log):
    budget = FT_STEPS * len(CHECKPOINTS)
    model, curve = replay(fresh(ctx), res.events, res.parents, H0,
                          ctx["layer"].eps, ctx["val"])
    v0 = model.loss(*ctx["val"])
    ft_block(model, ctx["train_ids"], steps=budget, lr=FT_LR,
             seed=block_seed(0))
    log.add("end", mode, curve[-1][0], budget, FT_LR, v0,
            model.loss(*ctx["val"]))
    curve.append((curve[-1][0], model.loss(*ctx["val"])))
    return curve


def baseline_orders(ctx):
    caps0 = ctx["layer"].capacities()
    model = ctx["model"]
    return dict(
        capacity=list(np.argsort(caps0)),
        l1=list(np.argsort(np.abs(model.W1).sum(axis=0))),
        random=list(np.random.default_rng(5).permutation(H0)),
    )


def run_arms(ctx):
    log, drift, curves = FTLog(), [], {}
    for mode in MODES:
        res = load_base(mode)
        curves[f"base_{mode}"] = res.val_curve
        print(f"arm A ({mode})")
        curves[f"A_{mode}"] = arm_a(ctx, mode, res, log)
        print(f"arm B ({mode})")
        curves[f"B_{mode}"] = arm_b(ctx, mode, log, drift)
        print(f"end-only ({mode})")
        curves[f"end_{mode}"] = arm_end(ctx, mode, res, log)

    for name, order in baseline_orders(ctx).items():
        print(f"baseline {name}: static and recovered")
        _, curves[f"{name}_static"] = replay_prune_order(
            fresh(ctx), order, H0, ctx["val"], TARGET_DENSITY)
        state = dict(idx=0)

        def hook(model, density, step, _name=name, _state=state):
            v0 = model.loss(*ctx["val"])
            ft_block(model, ctx["train_ids"], steps=FT_STEPS, lr=FT_LR,
                     seed=block_seed(_state["idx"]))
            log.add(_name, "static", density, FT_STEPS, FT_LR, v0,
                    model.loss(*ctx["val"]))
            _state["idx"] += 1
            return model

        _, curves[f"{name}_ft"] = replay_prune_order(
            fresh(ctx), order, H0, ctx["val"], TARGET_DENSITY,
            checkpoints=CHECKPOINTS, hook=hook)

    with open(os.path.join(OUT, "curves.pkl"), "wb") as fh:
        pickle.dump(dict(curves=curves, ft_rows=log.rows, drift=drift), fh)
    with open(os.path.join(OUT, "ft_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "mode", "density", "steps",
                                           "lr", "val_before", "val_after"])
        w.writeheader()
        w.writerows(log.rows)


def make_figures(curves, drift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        series = [(f"base_{mode}", "base (no recovery)"),
                  (f"A_{mode}", "A: frozen plan + recovery"),
                  (f"B_{mode}", "B: re-planned + recovery"),
                  (f"end_{mode}", "end-only recovery"),
                  ("l1_ft", "L1 + recovery"),
                  ("random_ft", "random + recovery")]
        cmap = plt.get_cmap("tab10")
        for k, (key, lab) in enumerate(series):
            c = curves[key]
            ax.plot([d for d, _ in c], [l for _, l in c],
                    label=lab, color=cmap(k), lw=1.6)
        ax.set_xlabel("density")
        ax.set_title(f"{mode} correlations")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation NLL (nats)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_recovery_trajectories.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for mode, marker in zip(MODES, ("o", "s")):
        rows = [r for r in drift if r["mode"] == mode]
        ax.plot([r["density"] for r in rows], [r["spearman"] for r in rows],
                marker=marker, label=f"{mode}")
    ax.set_xlabel("density at recovery block")
    ax.set_ylabel("Spearman(capacities pre, post recovery)")
    ax.invert_xaxis()
    ax.set_ylim(None, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_capacity_drift.png", dpi=140)
    plt.close(fig)


def order_divergence(base_res, b_res):
    rem_base = removed_originals(base_res.events, H0)
    rem_b = removed_originals(b_res.events, H0)
    common = sorted(set(rem_base) & set(rem_b))
    jaccard = len(common) / len(set(rem_base) | set(rem_b))
    rho = sps.spearmanr([rem_base[u] for u in common],
                        [rem_b[u] for u in common]).statistic \
        if len(common) > 2 else float("nan")
    return dict(n_base=len(rem_base), n_b=len(rem_b), n_common=len(common),
                jaccard=jaccard, spearman=float(rho))


def run_finish():
    with open(os.path.join(OUT, "curves.pkl"), "rb") as fh:
        blob = pickle.load(fh)
    curves, ft_rows, drift = blob["curves"], blob["ft_rows"], blob["drift"]
    make_figures(curves, drift)

    probe_densities = (0.60, 0.50, 0.42, TARGET_DENSITY)
    lines = ["# Stage 4 results: trajectory fine-tuning", ""]
    lines += [f"Recovery blocks: {FT_STEPS} Adam steps at lr {FT_LR} at each "
              f"of {len(CHECKPOINTS)} density checkpoints "
              f"({CHECKPOINTS[0]} down to {CHECKPOINTS[-1]}); end-only gets "
              f"the identical total budget of {FT_STEPS * len(CHECKPOINTS)} "
              "steps in one block. Recovery seeds are paired across arms.", ""]

    for mode in MODES:
        lines += [f"## {mode} correlations", "",
                  "| arm | " + " | ".join(f"val @ {d}" for d in probe_densities)
                  + " |",
                  "|---|" + "---|" * len(probe_densities)]
        keys = [(f"base_{mode}", "base"), (f"A_{mode}", "A: frozen plan"),
                (f"B_{mode}", "B: re-planned"), (f"end_{mode}", "end-only"),
                ("capacity_ft", "capacity + recovery"),
                ("l1_ft", "L1 + recovery"), ("random_ft", "random + recovery"),
                ("l1_static", "L1 static"), ("random_static", "random static")]
        for key, label in keys:
            vals = " | ".join(f"{loss_at(curves[key], d):.4f}"
                              for d in probe_densities)
            lines.append(f"| {label} | {vals} |")
        lines.append("")

        div = order_divergence(load_base(mode),
                               pickle.load(open(os.path.join(
                                   OUT, f"armB_{mode}.pkl"), "rb")))
        lines += [f"Eviction-set divergence, base vs re-planned ({mode}): "
                  f"Jaccard {div['jaccard']:.3f} over removed originals "
                  f"({div['n_common']} common of {div['n_base']}/{div['n_b']}),"
                  f" removal-order Spearman {div['spearman']:.3f} on the "
                  "common set.", ""]

    lines += ["## Capacity drift at recovery blocks", ""]
    for mode in MODES:
        rows = [r for r in drift if r["mode"] == mode]
        if rows:
            worst = min(rows, key=lambda r: r["spearman"])
            lines.append(f"- {mode}: Spearman(pre, post) ranges "
                         f"{worst['spearman']:.3f} (at density "
                         f"{worst['density']}) to "
                         f"{max(r['spearman'] for r in rows):.3f}.")
    lines += ["", "Figures: fig8_recovery_trajectories.png, "
              "fig9_capacity_drift.png."]

    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(part="all"):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if part in ("base", "pilot", "arms", "all"):
        ctx = build_context()
        assert ctx["model"].W1.shape[1] == H0
    if part in ("base", "all"):
        run_base(ctx)
    if part == "pilot":
        run_pilot(ctx)
    if part in ("arms", "all"):
        run_arms(ctx)
    if part in ("finish", "all"):
        run_finish()
    print(f"part {part} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
