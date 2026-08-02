"""Draw the map on GPT-2 -ßHOPE encoding of chosen MLP blocks, with GELU."""

from __future__ import annotations
import argparse
import csv
import os
import pickle
import time
from collections import defaultdict
import numpy as np
from scipy import stats as sps
from minihope.gelu_kernels import act_capacity, act_pair_kernel, act_self_kernel, gelu
from minihope.kernels import warped_correlation
from broad.run_broad import get_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

MODEL_NAME = "gpt2"
WINDOW = 1024
CALIB_WINDOWS = 8         
EVAL_WINDOWS = 16          # loss checkpoints
REG_WINDOWS = 24           # perregister calibration volume
BATCH = 2
EVAL_EVERY = 64            # actions between loss checkpoints
RHO_MIN, MAX_PAIRS = 0.10, 300
TARGET_DENSITY = 0.35
BLOCKS = (2, 6, 12)        
SEED = 0


def device_of():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.loss_type = "ForCausalLM" 
    model.eval().to(device_of())
    return tok, model


def sync_nf(mlp):
    mlp.c_fc.nf = mlp.c_fc.weight.shape[1]
    mlp.c_proj.nf = mlp.c_proj.weight.shape[1]


def blocks_of(model):
    return list(model.transformer.h)


def token_cache(corpus_name, tok, corpus, smoke=False):
    """BPE-encode train/val text and per register train text - cache npz"""
    path = os.path.join(OUT, f"tokens_{corpus_name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return dict(z)
    def enc(spans, cap):
        text = "\n\n".join(corpus.text[a:b] for _, a, b in spans)[:cap]
        ids = []
        for s0 in range(0, len(text), 400_000):     
            ids.extend(tok(text[s0:s0 + 400_000])["input_ids"]) 
        return np.array(ids, dtype=np.int32)
    cap = 400_000 if smoke else 6_000_000       
    out = {"train": enc(corpus.train_docs, cap),
           "val": enc(corpus.val_docs, cap)}
    genres = defaultdict(list)
    for d in corpus.train_docs:
        genres[d[0]].append(d)
    for g, spans in genres.items():
        vols = sum(b - a for _, a, b in spans)
        if vols >= 200_000:
            out[f"reg_{g}"] = enc(spans, 1_500_000)
    np.savez_compressed(path, **out)
    return out


def windows_from(ids, n, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - WINDOW - 1, size=n)
    return np.stack([ids[s:s + WINDOW] for s in starts]).astype(np.int64)


def forward_batches(model, windows, hooks=(), want_loss=False):
    import torch
    dev = next(model.parameters()).device
    handles = [m.register_forward_hook(fn) for m, fn in hooks]
    loss_sum = ntok = 0
    try:
        with torch.no_grad():
            for s in range(0, windows.shape[0], BATCH):
                x = torch.from_numpy(windows[s:s + BATCH]).to(dev)
                if want_loss:
                    out = model(x, labels=x)
                    k = x.shape[0] * (x.shape[1] - 1)
                    loss_sum += float(out.loss) * k
                    ntok += k
                else:
                    model(x)
    finally:
        for h in handles:
            h.remove()
    return loss_sum / ntok if ntok else None


def to2d(t):
    import torch
    return t.detach().to("cpu", torch.float32).numpy().reshape(-1, t.shape[-1])


def check_capture_and_writeback(model, win):
    import torch
    blk = blocks_of(model)[0]
    grabbed = {}
    def hook(mod, inp, out):
        grabbed["x"], grabbed["h"] = to2d(inp[0]), to2d(out)
    h = blk.mlp.c_fc.register_forward_hook(hook)
    with torch.no_grad():
        model(torch.from_numpy(win[None]).to(next(model.parameters()).device))
    h.remove()
    W = blk.mlp.c_fc.weight.detach().cpu().numpy()
    b = blk.mlp.c_fc.bias.detach().cpu().numpy()
    err1 = float(np.abs(grabbed["x"] @ W + b - grabbed["h"]).max())

    mlp = blk.mlp
    x = torch.from_numpy(grabbed["x"][:64]).to(next(model.parameters()).device)
    with torch.no_grad():
        full = mlp(x).cpu().numpy()
        keep = list(range(1, W.shape[1]))
        wfc, bfc = mlp.c_fc.weight.data.clone(), mlp.c_fc.bias.data.clone()
        wpr = mlp.c_proj.weight.data.clone()
        mlp.c_fc.weight.data = wfc[:, keep].contiguous()
        mlp.c_fc.bias.data = bfc[keep].contiguous()
        mlp.c_proj.weight.data = wpr[keep, :].contiguous()
        sync_nf(mlp)
        pruned = mlp(x).cpu().numpy()
        mlp.c_fc.weight.data, mlp.c_fc.bias.data = wfc, bfc
        mlp.c_proj.weight.data = wpr
        sync_nf(mlp)
    h0 = grabbed["x"][:64] @ W[:, 0] + b[0]
    manual = full - np.outer(gelu(h0), wpr[0].cpu().numpy())
    err2 = float(np.abs(pruned - manual).max())
    assert err1 < 1e-3 and err2 < 1e-3, (err1, err2)
    print(f"self-checks: capture {err1:.2e}, writeback {err2:.2e}")

def calib_stats(model, calib_win, map_blocks):
    blks = blocks_of(model)
    sums = [None] * len(blks)
    xs = {b: [] for b in map_blocks}
    hooks = []
    def out_hook(k):
        def fn(mod, inp, out):
            h = to2d(out).astype(np.float64)
            if sums[k] is None:
                sums[k] = [np.zeros(h.shape[1]) for _ in range(4)] + [0]
            s = sums[k]
            s[0] += h.sum(0); s[1] += (h**2).sum(0)
            s[2] += (h**3).sum(0); s[3] += (h**4).sum(0); s[4] += h.shape[0]
            if k + 1 in xs:
                xs[k + 1].append(to2d(inp[0]))
        return fn
    for k, blk in enumerate(blks):
        hooks.append((blk.mlp.c_fc, out_hook(k)))
    forward_batches(model, calib_win, hooks=hooks)
    stats = []
    for s in sums:
        n = s[4]
        mu = s[0] / n
        var = np.maximum(s[1] / n - mu**2, 1e-12)
        m3 = s[2] / n - 3 * mu * s[1] / n + 2 * mu**3
        m4 = (s[3] / n - 4 * mu * s[2] / n + 6 * mu**2 * s[1] / n - 3 * mu**4)
        stats.append(dict(mu=mu, sigma=np.sqrt(var),
                          exkurt=m4 / var**2 - 3.0,
                          skew=m3 / var**1.5))
    X = {b: np.concatenate(xs[b], axis=0) for b in map_blocks}
    return stats, X


class BlockState:

    def __init__(self, model, block_ix, mu, sigma, X):
        blk = blocks_of(model)[block_ix - 1]
        self.W = blk.mlp.c_fc.weight.detach().cpu().numpy().copy()   
        self.b = blk.mlp.c_fc.bias.detach().cpu().numpy().copy()
        self.Wo = blk.mlp.c_proj.weight.detach().cpu().numpy().copy() 
        self.mu, self.sigma = mu.copy(), sigma.copy()
        self.X = X                                  
        self.H = X @ self.W + self.b                  
        self.ids = np.arange(self.W.shape[1])
        self.next_id = self.W.shape[1]

    @property
    def n(self):
        return self.W.shape[1]

    def caps(self):
        return act_capacity(np.linalg.norm(self.Wo, axis=1), self.mu, self.sigma)

    def wt(self, i):                                   # augmented input vector
        return np.concatenate([self.W[:, i], [self.b[i]]])

    def drop(self, idx):
        keep = [k for k in range(self.n) if k not in set(idx)]
        for name in ("mu", "sigma", "b", "ids"):
            setattr(self, name, getattr(self, name)[keep])
        self.W = self.W[:, keep]; self.Wo = self.Wo[keep, :]
        self.H = self.H[:, keep]

    def add_parent(self, w, b, wo):
        h = self.X @ w + b
        self.W = np.concatenate([self.W, w[:, None]], axis=1)
        self.b = np.concatenate([self.b, [b]])
        self.Wo = np.concatenate([self.Wo, wo[None, :]], axis=0)
        self.mu = np.concatenate([self.mu, [h.mean()]])
        self.sigma = np.concatenate([self.sigma, [h.std() + 1e-12]])
        self.H = np.concatenate([self.H, h[:, None]], axis=1)
        self.ids = np.concatenate([self.ids, [self.next_id]])
        self.next_id += 1

    def push_to(self, model, block_ix):
        import torch
        blk = blocks_of(model)[block_ix - 1]
        dev = next(model.parameters()).device
        blk.mlp.c_fc.weight.data = torch.from_numpy(self.W).to(dev, torch.float32).contiguous()
        blk.mlp.c_fc.bias.data = torch.from_numpy(self.b).to(dev, torch.float32).contiguous()
        blk.mlp.c_proj.weight.data = torch.from_numpy(self.Wo).to(dev, torch.float32).contiguous()
        sync_nf(blk.mlp)


def warp_matrix(cos, r):
    cos = np.clip(cos, -0.999999, 0.999999)
    kappa = cos / (1.0 - cos**2) * np.outer(r, r)
    return 2.0 * kappa / (1.0 + np.sqrt(1.0 + 4.0 * kappa**2))


class CorrTracker:
    """Incrementally maintained pairwise structure over the active units"""

    def __init__(self, mode, W=None, H=None):
        self.mode = mode
        if mode == "paper":
            self.G = W.T @ W
        else:
            Z = H - H.mean(0)
            Z /= (np.linalg.norm(Z, axis=0) + 1e-12)
            self.Z = Z
            self.G = Z.T @ Z

    def matrix(self, st):
        if self.mode == "paper":
            wn = np.linalg.norm(st.W, axis=0) + 1e-12
            cos = self.G / np.outer(wn, wn)
            rho = warp_matrix(cos, st.sigma / wn)
        else:
            rho = np.clip(self.G, -1.0, 1.0)
        np.fill_diagonal(rho, 1.0)
        return rho

    def drop(self, idx):
        keep = np.setdiff1d(np.arange(self.G.shape[0]), idx)
        self.G = self.G[np.ix_(keep, keep)]
        if self.mode == "empirical":
            self.Z = self.Z[:, keep]

    def add_last_of(self, st):
        n = self.G.shape[0]
        if self.mode == "paper":
            w = st.W[:, -1]
            v = st.W[:, :n].T @ w if n else np.empty(0)
            d = float(w @ w)
        else:
            z = st.H[:, -1] - st.H[:, -1].mean()
            z /= (np.linalg.norm(z) + 1e-12)
            v = self.Z.T @ z if n else np.empty(0)
            self.Z = np.concatenate([self.Z, z[:, None]], axis=1)
            d = 1.0
        G = np.empty((n + 1, n + 1))
        G[:n, :n] = self.G
        G[:n, n] = v; G[n, :n] = v; G[n, n] = d
        self.G = G


def price_merge(st, i, j, rho_ij, e_a, caps):
    wt_i, wt_j = st.wt(i), st.wt(j)
    wo_i, wo_j = st.Wo[i], st.Wo[j]
    g_in = np.array([[wt_i @ wt_i, wt_i @ wt_j], [wt_j @ wt_i, wt_j @ wt_j]])
    g_out = np.array([[wo_i @ wo_i, wo_i @ wo_j], [wo_j @ wo_i, wo_j @ wo_j]])
    evals, evecs = np.linalg.eig(g_out @ g_in)
    alpha = np.real(evecs[:, int(np.argmax(np.real(evals)))])
    u = alpha[0] * wt_i + alpha[1] * wt_j
    if np.linalg.norm(u) < 1e-10:
        return None
    kij = float(act_pair_kernel([st.mu[i]], [st.sigma[i]], [st.mu[j]],
                                [st.sigma[j]], [rho_ij])[0])
    kii = float(act_self_kernel([st.mu[i]], [st.sigma[i]])[0])
    kjj = float(act_self_kernel([st.mu[j]], [st.sigma[j]])[0])
    a_val = (wo_i @ wo_i) * kii + (wo_j @ wo_j) * kjj + 2 * (wo_i @ wo_j) * kij

    best = None
    for sign in (1.0, -1.0):
        u_hat = sign * u / np.linalg.norm(u)
        w_p, b_p = u_hat[:-1], float(u_hat[-1])
        h_p = st.X @ w_p + b_p
        mu_p, sg_p = float(h_p.mean()), float(h_p.std() + 1e-12)
        kpp = float(act_self_kernel([mu_p], [sg_p])[0])
        if kpp < 1e-12:
            continue
        rho_pi = float(np.corrcoef(h_p, st.H[:, i])[0, 1])
        rho_pj = float(np.corrcoef(h_p, st.H[:, j])[0, 1])
        kpi = float(act_pair_kernel([mu_p], [sg_p], [st.mu[i]], [st.sigma[i]],
                                    [rho_pi])[0])
        kpj = float(act_pair_kernel([mu_p], [sg_p], [st.mu[j]], [st.sigma[j]],
                                    [rho_pj])[0])
        v = kpi * wo_i + kpj * wo_j
        b_val = float(np.linalg.norm(v)) / np.sqrt(kpp)
        if best is None or b_val > best["b"]:
            best = dict(b=b_val, w=w_p, bias=b_p, mu=mu_p, sg=sg_p, kpp=kpp,
                        v=v)
    if best is None or best["b"] <= 0:
        return None
    e_rem = max(e_a - caps[i] - caps[j], 1e-12)
    a, b_val = a_val, best["b"]
    s_star = (a + b_val * e_rem) / (2.0 * e_rem + b_val)
    d = np.sqrt(max(a + 2 * s_star**2 - 2 * s_star * b_val, 0.0))
    cost = st.n * d / max(e_rem + s_star, 1e-12)
    v_hat = best["v"] / (np.linalg.norm(best["v"]) + 1e-12)
    wo_p = (s_star / np.sqrt(best["kpp"])) * v_hat
    return dict(cost=float(cost), w=best["w"], bias=best["bias"], wo=wo_p,
                rho=float(rho_ij))


def encode_block(model, block_ix, stats, X, mode, val_win, target_density,
                 out_prefix):
    st = BlockState(model, block_ix, stats["mu"], stats["sigma"], X)
    h0 = st.n
    tracker = (CorrTracker("paper", W=st.W) if mode == "paper"
               else CorrTracker("empirical", H=st.H))
    events, curve = [], []
    st.push_to(model, block_ix)
    curve.append((1.0, forward_batches(model, val_win, want_loss=True)))
    step = 0
    t0 = time.time()
    while st.n / h0 > target_density:
        caps = st.caps()
        e_a = float(caps.sum())
        rho = tracker.matrix(st)
        assert rho.shape[0] == st.n, "tracker desynchronized from state"
        j_prune = st.n * caps / np.maximum(e_a - caps, 1e-12)
        best = dict(action="prune", i=int(np.argmin(j_prune)),
                    cost=float(j_prune.min()))
        iu = np.triu_indices(st.n, 1)
        cand = np.where(rho[iu] >= RHO_MIN)[0]
        if cand.size > MAX_PAIRS:
            cand = cand[np.argsort(rho[iu][cand])[::-1][:MAX_PAIRS]]
        for c in cand:
            i, j = int(iu[0][c]), int(iu[1][c])
            m = price_merge(st, i, j, rho[i, j], e_a, caps)
            if m and m["cost"] < best["cost"]:
                best = dict(action="merge", i=i, j=j, **m)
        if best["action"] == "prune":
            i = best["i"]
            events.append(("prune", int(st.ids[i]), None, float(caps[i]),
                           best["cost"]))
            st.drop([i])
            tracker.drop([i])
        else:
            i, j = best["i"], best["j"]
            events.append(("merge", int(st.ids[i]), int(st.ids[j]),
                           float(caps[i] + caps[j]), best["cost"]))
            st.add_parent(best["w"], best["bias"], best["wo"])
            tracker.add_last_of(st)
            st.drop([i, j])
            tracker.drop([i, j])
        step += 1
        if step % EVAL_EVERY == 0 or st.n / h0 <= target_density:
            st.push_to(model, block_ix)
            nll = forward_batches(model, val_win, want_loss=True)
            curve.append((st.n / h0, nll))
            print(f"  [b{block_ix} {mode}] step {step:>5} N {st.n:>5} "
                  f"density {st.n/h0:.3f} nll {nll:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(f"{out_prefix}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["action", "id_i", "id_j", "cap", "cost"])
        w.writerows(events)
    np.savez(f"{out_prefix}.npz", curve=np.array(curve),
             removed=np.array([e[1] for e in events] +
                              [e[2] for e in events if e[2] is not None]))
    return events, curve


def restore(model, snapshot):
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data = snapshot[name].clone()
    for blk in blocks_of(model):
        sync_nf(blk.mlp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all",
                    choices=["prep", "calib", "map", "registers", "finish", "all"])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--blocks", type=int, nargs="+", default=list(BLOCKS))
    ap.add_argument("--density", type=float, default=TARGET_DENSITY)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = os.path.join(HERE, "results_smoke")
    os.makedirs(OUT, exist_ok=True)
    density = 0.9 if args.smoke else args.density
    t0 = time.time()

    corpus = get_corpus(args.corpus)
    tok, model = load_model()
    toks = token_cache(corpus.name, tok, corpus, smoke=args.smoke)
    calib_win = windows_from(toks["train"], 2 if args.smoke else CALIB_WINDOWS, 10)
    val_win = windows_from(toks["val"], 4 if args.smoke else EVAL_WINDOWS, 0)
    check_capture_and_writeback(model, calib_win[0])

    spath = os.path.join(OUT, "calib_stats.pkl")
    if args.part in ("all", "prep", "calib") or not os.path.exists(spath):
        stats, X = calib_stats(model, calib_win, args.blocks)
        pickle.dump(dict(stats=stats, X=X, blocks=args.blocks),
                    open(spath, "wb"))
        base = forward_batches(model, val_win, want_loss=True)
        print(f"base nll {base:.4f}; per-block median exkurt:",
              [round(float(np.median(s["exkurt"])), 2) for s in stats])
        if args.part in ("prep", "calib"):
            return
    z = pickle.load(open(spath, "rb"))
    stats, X = z["stats"], z["X"]

    if args.part in ("all", "map"):
        import copy
        snapshot = copy.deepcopy(model.state_dict())
        for bix in args.blocks:
            if bix not in X:
                raise SystemExit(f"block {bix} has no cached X; rerun calib "
                                 f"with --blocks {' '.join(map(str, args.blocks))}")
            for mode in ("paper", "empirical"):
                pref = os.path.join(OUT, f"map_b{bix}_{mode}")
                if os.path.exists(pref + ".npz"):
                    print(f"skip existing {pref}")
                    continue
                print(f"encoding block {bix} [{mode}] to {density:.0%}")
                encode_block(model, bix, stats[bix - 1], X[bix], mode,
                             val_win, density, pref)
                restore(model, snapshot)
            st = BlockState(model, bix, stats[bix - 1]["mu"],
                            stats[bix - 1]["sigma"], X[bix])
            orders = {"capacity": np.argsort(st.caps()),
                      "l1": np.argsort(np.abs(st.W).sum(0)),
                      "random": np.random.default_rng(3).permutation(st.n)}
            for name, order in orders.items():
                pref = os.path.join(OUT, f"map_b{bix}_base_{name}")
                if os.path.exists(pref + ".npz"):
                    continue
                st = BlockState(model, bix, stats[bix - 1]["mu"],
                                stats[bix - 1]["sigma"], X[bix])
                keep_curve = [(1.0, forward_batches(model, val_win, want_loss=True))]
                h0_b = st.n
                alive = list(range(st.n))
                for cnt, orig in enumerate(order, 1):
                    st.drop([alive.index(orig)])
                    alive.remove(orig)
                    if cnt % EVAL_EVERY == 0 or len(alive) / h0_b <= density:
                        st.push_to(model, bix)
                        keep_curve.append((len(alive) / h0_b,
                                           forward_batches(model, val_win,
                                                           want_loss=True)))
                    if len(alive) / h0_b <= density:
                        break
                np.savez(pref + ".npz", curve=np.array(keep_curve))
                restore(model, snapshot)
                print(f"  baseline {name} b{bix} done")

    if args.part in ("all", "registers"):
        regs = sorted(k for k in toks if k.startswith("reg_"))
        rows = {}
        for rk in regs:
            win = windows_from(toks[rk], 4 if args.smoke else REG_WINDOWS,
                               100 + len(rows))
            rstats, _ = calib_stats(model, win, [])
            rows[rk[4:]] = np.stack([
                act_capacity(np.linalg.norm(
                    blocks_of(model)[k].mlp.c_proj.weight.detach().cpu().numpy(),
                    axis=1), s["mu"], s["sigma"])
                for k, s in enumerate(rstats)])          # (12, 3072)
            print(f"register {rk[4:]} calibrated")
        pickle.dump(rows, open(os.path.join(OUT, "register_caps.pkl"), "wb"))

    if args.part in ("all", "finish"):
        rows = pickle.load(open(os.path.join(OUT, "register_caps.pkl"), "rb"))
        names = sorted(rows)
        depth_med = []
        for k in range(12):
            mats = np.stack([rows[n][k] for n in names])
            iu = np.triu_indices(len(names), 1)
            sp = [sps.spearmanr(mats[a], mats[b]).statistic
                  for a, b in zip(*iu)]
            depth_med.append(float(np.median(sp)))
        kurt_depth = [float(np.median(st_["exkurt"])) for st_ in stats]

        def curve_info(path):
            z = np.load(path)
            c = z["curve"]
            return c, float(c[-1, 0]), float(c[0, 1]), float(c[-1, 1])

        lines, div, curves_by_block = [], [], {}
        for bix in args.blocks:
            blk_lines = [f"## block {bix} (median exkurt "
                         f"{kurt_depth[bix - 1]:+.2f})"]
            counts = {}
            for mode in ("paper", "empirical"):
                with open(os.path.join(OUT, f"map_b{bix}_{mode}.csv")) as fh:
                    acts = [row.split(",")[0] for row in fh][1:]
                counts[mode] = (acts.count("prune"), acts.count("merge"))
                c, dfin, nll0, nll1 = curve_info(
                    os.path.join(OUT, f"map_b{bix}_{mode}.npz"))
                curves_by_block.setdefault(bix, {})[mode] = c
                blk_lines.append(
                    f"- {mode}: {counts[mode][0]} prunes, "
                    f"{counts[mode][1]} merges; NLL {nll0:.4f} -> {nll1:.4f} "
                    f"at density {dfin:.2f} ({c.shape[0]} checkpoints)")
            for base in ("capacity", "l1", "random"):
                p = os.path.join(OUT, f"map_b{bix}_base_{base}.npz")
                if os.path.exists(p):
                    c, dfin, nll0, nll1 = curve_info(p)
                    curves_by_block[bix][base] = c
                    blk_lines.append(f"- baseline {base}: NLL {nll0:.4f} -> "
                                     f"{nll1:.4f} at density {dfin:.2f}")
            rp = np.load(os.path.join(OUT, f"map_b{bix}_paper.npz"))
            re_ = np.load(os.path.join(OUT, f"map_b{bix}_empirical.npz"))
            sp_, se_ = set(rp["removed"].tolist()), set(re_["removed"].tolist())
            jac = len(sp_ & se_) / max(len(sp_ | se_), 1)
            div.append((bix, kurt_depth[bix - 1], jac))
            blk_lines.append(f"- paper-vs-empirical removed-set Jaccard {jac:.3f}")
            lines += blk_lines + [""]

        if len(div) > 2 and len({round(d[2], 6) for d in div}) > 1:
            r = sps.spearmanr([d[1] for d in div], [d[2] for d in div]).statistic
            lines.append(f"- rank correlation, non-Gaussianity vs map "
                         f"divergence across blocks: {r:+.2f}")
        else:
            lines.append("- divergence has no variance across blocks (the two "
                         "correlation models produced identical maps), so the "
                         "kurtosis-tracking hypothesis is moot: rho never "
                         "changed a decision. See the merge counts above.")
        lines.append("- cross-register capacity Spearman (median) by block: " +
                     ", ".join(f"b{k+1} {v:.3f}" for k, v in enumerate(depth_med)))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(curves_by_block),
                                 figsize=(4.4 * len(curves_by_block), 3.6),
                                 squeeze=False)
        for ax, (bix, cs) in zip(axes[0], sorted(curves_by_block.items())):
            for name, c in cs.items():
                ax.plot(c[:, 0], c[:, 1],
                        lw=1.7 if name in ("paper", "empirical") else 1.0,
                        label=name)
            ax.invert_xaxis(); ax.set_title(f"block {bix}")
            ax.set_xlabel("block density"); ax.set_ylabel("NLL")
            ax.legend(fontsize=7)
        fig.suptitle("GPT-2: per-block compression trajectories")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_g1_trajectories.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        ax.plot(range(1, 13), depth_med, marker="o", color="tab:blue",
                label="cross-register capacity Spearman (median)")
        ax.set_xlabel("block"); ax.set_ylabel("Spearman", color="tab:blue")
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        ax2.plot(range(1, 13), kurt_depth, marker="s", ls="--",
                 color="tab:red", label="median excess kurtosis")
        ax2.set_ylabel("excess kurtosis", color="tab:red")
        ax.set_title("Register-universality and non-Gaussianity by depth")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_g2_register_depth.png"), dpi=140)
        plt.close(fig)

        text = "# GPT-2 map results\n\n" + "\n".join(lines) + "\n"
        open(os.path.join(OUT, "results_gpt2_map.md"), "w").write(text)
        print(text)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()