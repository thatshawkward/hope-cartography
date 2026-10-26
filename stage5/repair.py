"""repair mechanisms for GPT-2 trajectory fine-tuning

Three ways to heal a wounded block, answering different questions:
  reproject   closed-form c_proj refit so survivors mimic the ORIGINAL block
              output on calibration activations - pure redundancy, no training
  block SGD   end-to-end loss, only the wounded block's MLP updates - can the
              block re-express itself?
  full SGD    the whole model updates - other blocks may compensate
"""

from __future__ import annotations

import numpy as np

from gpt2map.run_gpt2_map import BATCH, WINDOW, blocks_of
from minihope.gelu_kernels import gelu


def block_output(st):
    return gelu(st.H) @ st.Wo


def reproject(st, y_target):
    a = gelu(st.H)
    st.Wo = np.linalg.lstsq(a, y_target, rcond=None)[0].astype(st.Wo.dtype)


def pull_from(st, model, block_ix):
    """sync the numpy state from the live block after SGD moved it;
    mu/sigma are re-measured on the cached calibration inputs"""
    blk = blocks_of(model)[block_ix - 1]
    st.W = blk.mlp.c_fc.weight.detach().cpu().numpy().copy()
    st.b = blk.mlp.c_fc.bias.detach().cpu().numpy().copy()
    st.Wo = blk.mlp.c_proj.weight.detach().cpu().numpy().copy()
    st.H = st.X @ st.W + st.b
    st.mu = st.H.mean(0)
    st.sigma = st.H.std(0) + 1e-12


def sgd_recover(model, train_toks, steps, lr, seed, scope, block_ix=None):
    """scope 'full' or 'block'. Runs in eval mode: dropout off, so recovery is
    deterministic and paired seeds are meaningful."""
    import torch
    dev = next(model.parameters()).device
    saved = None
    if scope == "block":
        saved = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        blk = blocks_of(model)[block_ix - 1]
        train_params = [blk.mlp.c_fc.weight, blk.mlp.c_fc.bias,
                        blk.mlp.c_proj.weight, blk.mlp.c_proj.bias]
        for p in train_params:
            p.requires_grad_(True)
    else:
        train_params = list(model.parameters())
    opt = torch.optim.AdamW(train_params, lr=lr)
    rng = np.random.default_rng(seed)
    hi = len(train_toks) - WINDOW - 1
    for _ in range(steps):
        starts = rng.integers(0, hi, size=BATCH)
        xb = np.stack([train_toks[s:s + WINDOW]
                       for s in starts]).astype(np.int64)
        x = torch.from_numpy(xb).to(dev)
        out = model(x, labels=x)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    if saved is not None:
        for p, r in zip(model.parameters(), saved):
            p.requires_grad_(r)
