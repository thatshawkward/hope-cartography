"""Run Stage 5: trajectory fine-tuning on GPT-2 blocks.

Usage:  PYTHONPATH=src:. python -m stage5.run_stage5 [part] [--blocks 2 12]
        part  prep | verify | pilot | arms | finish | all
Writes curves, logs, figures, and results into stage5/results/.

The existing gpt2map runs contain zero merges, so every recorded map is a
pure prune order: replayable directly from the tracked CSVs, with the
tracked npz curves serving as the no-recovery baselines. Plans x repairs:
  plans    hope (recorded order, frozen) | random (seed-3 order, the
           published baseline) | replan (re-rank survivors by capacity
           after each repair, the greedy-under-zero-merges equivalent
           of stage 4's arm B)
  repairs  reproject | blocksgd | fullsgd (see stage5.repair), plus an
           end-only fullsgd/reproject comparator at matched total budget
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import time

import numpy as np

from gpt2map.run_gpt2_map import (BLOCKS, CALIB_WINDOWS, EVAL_EVERY,
                                  EVAL_WINDOWS, BlockState, calib_stats,
                                  forward_batches, get_corpus, load_model,
                                  restore, token_cache, windows_from)
from stage4.run_stage4 import loss_at
from stage5.repair import block_output, pull_from, reproject, sgd_recover

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
GOUT = os.path.join(os.path.dirname(HERE), "gpt2map", "results")

CHECKPOINTS = (0.90, 0.75, 0.60, 0.45, 0.35)
TARGET = 0.35
FT_STEPS = 300          # per recovery block; pilot-confirmed
FT_LR = 1e-5
BLOCKS5 = (2, 12)


def load_ctx(blocks):
    corpus = get_corpus("oanc")
    tok, model = load_model()
    toks = token_cache(corpus.name, tok, corpus)
    calib_win = windows_from(toks["train"], CALIB_WINDOWS, 10)
    val_win = windows_from(toks["val"], EVAL_WINDOWS, 0)
    spath = os.path.join(OUT, "calib.pkl")
    if os.path.exists(spath):
        z = pickle.load(open(spath, "rb"))
        stats, X = z["stats"], z["X"]
        missing = [b for b in blocks if b not in X]
    else:
        stats, X, missing = None, None, list(blocks)
    if missing:
        stats, X = calib_stats(model, calib_win, list(blocks))
        pickle.dump(dict(stats=stats, X=X), open(spath, "wb"))
    snapshot = copy.deepcopy(model.state_dict())
    return dict(model=model, toks=toks, calib_win=calib_win, val_win=val_win,
                stats=stats, X=X, snapshot=snapshot)


def hope_order(bix):
    with open(os.path.join(GOUT, f"map_b{bix}_empirical.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["action"] == "prune" for r in rows), \
        f"block {bix} map contains merges; stage 5 replay assumes prunes only"
    return [int(r["id_i"]) for r in rows]


def fresh_state(ctx, bix):
    restore(ctx["model"], ctx["snapshot"])
    s = ctx["stats"][bix - 1]
    return BlockState(ctx["model"], bix, s["mu"], s["sigma"], ctx["X"][bix])


def make_repair(ctx, bix, kind, y_target, log, plan, steps=FT_STEPS):
    state = dict(idx=0)

    def fn(st, density):
        model = ctx["model"]
        st.push_to(model, bix)
        pre = forward_batches(model, ctx["val_win"], want_loss=True)
        if kind == "reproject":
            reproject(st, y_target)
            st.push_to(model, bix)
        else:
            scope = "block" if kind == "blocksgd" else "full"
            sgd_recover(model, ctx["toks"]["train"], steps=steps, lr=FT_LR,
                        seed=1000 + state["idx"], scope=scope, block_ix=bix)
            if plan == "replan" and scope == "full":
                # earlier blocks moved: the calibration inputs are stale
                _, Xn = calib_stats(model, ctx["calib_win"], [bix])
                st.X = Xn[bix]
            pull_from(st, model, bix)
        post = forward_batches(model, ctx["val_win"], want_loss=True)
        log.append(dict(block=bix, plan=plan, repair=kind, density=density,
                        steps=0 if kind == "reproject" else steps,
                        nll_pre=pre, nll_post=post))
        state["idx"] += 1
        return pre, post

    return fn


def run_replay(ctx, bix, order, repair, label):
    st = fresh_state(ctx, bix)
    model, val_win = ctx["model"], ctx["val_win"]
    h0 = st.n
    curve = [(1.0, forward_batches(model, val_win, want_loss=True))]
    cps = list(CHECKPOINTS)
    for cnt, u in enumerate(order, 1):
        st.drop([int(np.where(st.ids == u)[0][0])])
        density = st.n / h0
        if cnt % EVAL_EVERY == 0 or density <= TARGET:
            st.push_to(model, bix)
            curve.append((density, forward_batches(model, val_win,
                                                   want_loss=True)))
        while cps and density <= cps[0] + 1e-9:
            cps.pop(0)
            if repair is not None:
                pre, post = repair(st, density)
                curve.append((density, pre))
                curve.append((density, post))
        if density <= TARGET:
            break
    print(f"  [{label}] done, final nll {curve[-1][1]:.4f}", flush=True)
    return curve


def run_replan(ctx, bix, repair, label):
    st = fresh_state(ctx, bix)
    model, val_win = ctx["model"], ctx["val_win"]
    h0 = st.n
    curve = [(1.0, forward_batches(model, val_win, want_loss=True))]
    cps = list(CHECKPOINTS)
    cnt = 0
    while st.n / h0 > TARGET:
        st.drop([int(np.argmin(st.caps()))])
        cnt += 1
        density = st.n / h0
        if cnt % EVAL_EVERY == 0 or density <= TARGET:
            st.push_to(model, bix)
            curve.append((density, forward_batches(model, val_win,
                                                   want_loss=True)))
        while cps and density <= cps[0] + 1e-9:
            cps.pop(0)
            pre, post = repair(st, density)
            curve.append((density, pre))
            curve.append((density, post))
    print(f"  [{label}] done, final nll {curve[-1][1]:.4f}", flush=True)
    return curve


def save_curve(name, curve):
    np.savez(os.path.join(OUT, f"{name}.npz"), curve=np.array(curve))


def run_verify(ctx, blocks):
    for bix in blocks:
        for plan, order in (("empirical", hope_order(bix)),
                            ("base_random",
                             list(np.random.default_rng(3).permutation(
                                 3072)))):
            curve = run_replay(ctx, bix, order, None, f"verify b{bix} {plan}")
            ref = np.load(os.path.join(GOUT, f"map_b{bix}_{plan}.npz"))
            ref = ref["curve"]
            ours = np.array(curve)
            n = min(len(ours), len(ref))
            dmax = float(np.abs(ours[:n, 1] - ref[:n, 1]).max())
            print(f"b{bix} {plan}: {len(ours)} vs {len(ref)} points, "
                  f"max |nll delta| {dmax:.5f}")


def run_pilot(ctx, bix=2):
    order = hope_order(bix)
    st = fresh_state(ctx, bix)
    model, val_win = ctx["model"], ctx["val_win"]
    h0 = st.n
    for u in order:
        st.drop([int(np.where(st.ids == u)[0][0])])
        if st.n / h0 <= 0.60:
            break
    st.push_to(model, bix)
    v0 = forward_batches(model, val_win, want_loss=True)
    comp = copy.deepcopy(model.state_dict())
    print(f"pilot b{bix} at density {st.n / h0:.3f}, nll {v0:.4f}")
    print(f"{'steps':>6} {'lr':>8} {'nll':>8} {'recovered':>10}")
    for lr in (5e-6, 2e-5):
        for steps in (100, 300, 1000):
            restore(model, comp)
            sgd_recover(model, ctx["toks"]["train"], steps=steps, lr=lr,
                        seed=999, scope="full")
            v = forward_batches(model, val_win, want_loss=True)
            print(f"{steps:>6} {lr:>8.0e} {v:>8.4f} {v0 - v:>10.4f}",
                  flush=True)
    restore(model, ctx["snapshot"])


def run_arms(ctx, blocks):
    log = []
    for bix in blocks:
        order = hope_order(bix)
        rand = list(np.random.default_rng(3).permutation(3072))
        y_target = block_output(fresh_state(ctx, bix))
        for repair_kind in ("reproject", "blocksgd", "fullsgd"):
            for plan, runner, arg in (("hope", run_replay, order),
                                      ("random", run_replay, rand),
                                      ("replan", run_replan, None)):
                name = f"run_b{bix}_{plan}_{repair_kind}"
                if os.path.exists(os.path.join(OUT, f"{name}.npz")):
                    print(f"  skip existing {name}")
                    continue
                rep = make_repair(ctx, bix, repair_kind, y_target, log, plan)
                if plan == "replan":
                    curve = runner(ctx, bix, rep, name)
                else:
                    curve = runner(ctx, bix, arg, rep, name)
                save_curve(name, curve)
        for kind, steps in (("fullsgd", FT_STEPS * len(CHECKPOINTS)),
                            ("reproject", 0)):
            name = f"run_b{bix}_hope_end_{kind}"
            if os.path.exists(os.path.join(OUT, f"{name}.npz")):
                continue
            rep = make_repair(ctx, bix, kind, y_target, log, "end",
                              steps=steps)
            curve = run_replay(ctx, bix, order, None, name + " (compress)")
            st = fresh_state(ctx, bix)
            for u in order:
                st.drop([int(np.where(st.ids == u)[0][0])])
                if st.n / 3072 <= TARGET:
                    break
            pre, post = rep(st, st.n / 3072)
            curve += [(st.n / 3072, pre), (st.n / 3072, post)]
            save_curve(name, curve)
        with open(os.path.join(OUT, "repairs_log.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["block", "plan", "repair",
                                               "density", "steps", "nll_pre",
                                               "nll_post"])
            w.writeheader()
            w.writerows(log)


def curves_for(bix):
    out = {}
    for plan in ("empirical", "base_random", "base_l1"):
        z = np.load(os.path.join(GOUT, f"map_b{bix}_{plan}.npz"))
        out[f"static_{plan}"] = [tuple(r) for r in z["curve"]]
    for f in sorted(os.listdir(OUT)):
        if f.startswith(f"run_b{bix}_") and f.endswith(".npz"):
            key = f[len(f"run_b{bix}_"):-4]
            out[key] = [tuple(r) for r in
                        np.load(os.path.join(OUT, f))["curve"]]
    return out


def run_finish(blocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe = (0.60, 0.45, TARGET)
    lines = ["# Stage 5 results: trajectory fine-tuning on GPT-2", "",
             f"Recovery: {FT_STEPS} AdamW steps at lr {FT_LR} (dropout off) "
             f"at {len(CHECKPOINTS)} density checkpoints; re-projection is "
             "one closed-form c_proj refit per checkpoint, zero gradient "
             "steps; end-only gets the matched total SGD budget of "
             f"{FT_STEPS * len(CHECKPOINTS)} steps.", ""]
    series_label = {"static_empirical": "HOPE static (no recovery)",
                    "static_base_random": "random static",
                    "static_base_l1": "L1 static",
                    "hope_reproject": "HOPE + re-projection",
                    "hope_blocksgd": "HOPE + block SGD",
                    "hope_fullsgd": "HOPE + full SGD",
                    "random_fullsgd": "random + full SGD",
                    "replan_fullsgd": "re-planned + full SGD",
                    "replan_reproject": "re-planned + re-projection",
                    "replan_blocksgd": "re-planned + block SGD",
                    "random_reproject": "random + re-projection",
                    "random_blocksgd": "random + block SGD",
                    "hope_end_fullsgd": "end-only full SGD",
                    "hope_end_reproject": "end-only re-projection"}
    for bix in blocks:
        cv = curves_for(bix)
        lines += [f"## Block {bix}", "",
                  "| run | " + " | ".join(f"nll @ {d}" for d in probe) + " |",
                  "|---|" + "---|" * len(probe)]
        for key, lab in series_label.items():
            if key in cv:
                vals = " | ".join(f"{loss_at(cv[key], d):.4f}" for d in probe)
                lines.append(f"| {lab} | {vals} |")
        lines.append("")

        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        cmap = plt.get_cmap("tab10")
        show = ["static_empirical", "static_base_random", "hope_reproject",
                "hope_blocksgd", "hope_fullsgd", "random_fullsgd",
                "replan_fullsgd", "hope_end_fullsgd"]
        for k, key in enumerate(show):
            if key not in cv:
                continue
            c = cv[key]
            ax.plot([d for d, _ in c], [l for _, l in c],
                    label=series_label[key], color=cmap(k), lw=1.5,
                    ls="--" if "end" in key else "-")
        ax.set_xlabel("block density")
        ax.set_ylabel("validation NLL (nats)")
        ax.set_title(f"GPT-2 block {bix}: eviction with recovery")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_g{bix + 1}_b{bix}_recovery.png"),
                    dpi=140)
        plt.close(fig)
        lines.append(f"Figure: fig_g{bix + 1}_b{bix}_recovery.png")
        lines.append("")

    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all",
                    choices=["prep", "verify", "pilot", "arms", "finish",
                             "all"])
    ap.add_argument("--blocks", type=int, nargs="+", default=list(BLOCKS5))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ctx = None
    if args.part in ("prep", "verify", "pilot", "arms", "all"):
        ctx = load_ctx(args.blocks)
    if args.part in ("verify", "all"):
        run_verify(ctx, args.blocks)
    if args.part == "pilot":
        run_pilot(ctx)
    if args.part in ("arms", "all"):
        run_arms(ctx, args.blocks)
    if args.part in ("finish", "all"):
        run_finish(args.blocks)
    print(f"part {args.part} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
