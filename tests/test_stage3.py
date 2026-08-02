"""Stage 3 checks: writeback exactness and the rho override."""

import numpy as np
import pytest

from minihope.merge import merge_pair
from stage2.calibrate import calibrated_layer
from stage2.model import CharMLP
from stage3.writeback import effective_bias_of, merge_units, prune_unit


def _setup(seed=0, hidden=12):
    rng = np.random.default_rng(seed)
    model = CharMLP(vocab=20, k=4, d=6, hidden=hidden, seed=seed)
    ctx = rng.integers(0, 20, size=(4000, 4))
    tgt = rng.integers(0, 20, size=4000)
    layer = calibrated_layer(model, ctx)
    return model, layer, ctx, tgt


def test_prune_writeback_is_exact():
    model, layer, ctx, tgt = _setup()
    k = 5
    pruned = prune_unit(model, k)
    x = model.features(ctx)
    h = np.maximum(x @ model.W1 + model.b1, 0.0)
    manual = h @ model.W2 + model.b2 - np.outer(h[:, k], model.W2[k])
    got = np.maximum(x @ pruned.W1 + pruned.b1, 0.0) @ pruned.W2 + pruned.b2
    assert got == pytest.approx(manual, abs=1e-10)


def test_merge_writeback_realizes_the_parent():
    model, layer, ctx, tgt = _setup(seed=1)
    res = merge_pair(layer, 0, 1)
    merged = merge_units(model, 0, 1, res.parent, layer.eps)
    assert merged.W1.shape[1] == model.W1.shape[1] - 1
    x = model.features(ctx[:500])
    scale = res.parent["gamma"] / np.sqrt(res.parent["sigma"] ** 2 + layer.eps)
    expect = x @ (scale * res.parent["w_raw"]) + effective_bias_of(res.parent, layer.eps)
    got = x @ merged.W1[:, 0] + merged.b1[0]
    assert got == pytest.approx(expect, abs=1e-9)
    assert merged.W2[1:, :] == pytest.approx(model.W2[2:, :], abs=0)


def test_rho_override_matches_default():
    model, layer, ctx, tgt = _setup(seed=2)
    base = merge_pair(layer, 2, 7)
    over = merge_pair(layer, 2, 7, rho=layer.rho_hat(2, 7))
    assert over.cost == pytest.approx(base.cost, rel=1e-12)
    assert over.s_star == pytest.approx(base.s_star, rel=1e-12)
