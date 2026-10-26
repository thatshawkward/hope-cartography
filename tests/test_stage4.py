"""trajectory fine-tuning machinery: replay fidelity, hooks, budget accounting"""

import numpy as np
import pytest

from stage2.calibrate import calibrated_layer
from stage2.data import sample_contexts
from stage2.model import CharMLP
from stage3.encoder import encode_and_map
from stage3.writeback import clone_model, prune_unit
from stage4.finetune import FTLog, checkpoint_densities, ft_block
from stage4.replay import replay, replay_prune_order

VOCAB, K, HID = 11, 3, 16


def make_stream(n, rng):
    # learnable structure: next char is a lagged sum, with 10% noise
    ids = [1, 2]
    for _ in range(n - 2):
        nxt = (ids[-1] + ids[-2]) % VOCAB
        if rng.random() < 0.1:
            nxt = int(rng.integers(0, VOCAB))
        ids.append(nxt)
    return np.asarray(ids, dtype=np.int64)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    train_ids = make_stream(6000, rng)
    val_ids = make_stream(2000, rng)
    model = CharMLP(VOCAB, k=K, d=4, hidden=HID, seed=0)
    model.fit(train_ids, steps=400, batch=64, log_every=10**9)
    for dst in (1, 2):
        # exact duplicates of unit 0, so the empirical map must choose merges
        model.W1[:, dst] = model.W1[:, 0]
        model.b1[dst] = model.b1[0]
        model.W2[dst, :] = model.W2[0, :]
    val = sample_contexts(val_ids, K, 512, np.random.default_rng(1))
    calib, _ = sample_contexts(train_ids, K, 1024, np.random.default_rng(2))
    layer = calibrated_layer(model, calib)
    return dict(model=model, layer=layer, val=val,
                train_ids=train_ids, calib=calib)


def base_map(setup, **kw):
    args = dict(probe_ctx=setup["calib"], mode="empirical",
                target_density=0.5, rho_min=0.05, max_pairs=40,
                min_units=4, verbose_every=0)
    args.update(kw)
    return encode_and_map(clone_model(setup["model"]), setup["layer"],
                          ["u"] * HID, np.zeros(HID), setup["val"], **args)


def test_checkpoint_densities_grid():
    cps = checkpoint_densities()
    assert cps[0] == 0.95 and cps[-1] == 0.35 and len(cps) == 13
    assert all(a > b for a, b in zip(cps, cps[1:]))
    assert checkpoint_densities(stop=0.33)[-1] == 0.33


def test_replay_reproduces_base(setup):
    res = base_map(setup)
    assert any(e.action == "merge" for e in res.events), \
        "sequence exercises no merge; replay fidelity untested"
    _, curve = replay(clone_model(setup["model"]), res.events, res.parents,
                      HID, setup["layer"].eps, setup["val"])
    a = np.asarray(res.val_curve)
    b = np.asarray(curve)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_replay_hook_fires_at_checkpoints(setup):
    res = base_map(setup)
    seen = []

    def hook(model, density, step):
        seen.append(density)
        return ft_block(model, setup["train_ids"], steps=30, lr=3e-4,
                        seed=1, batch=32)

    _, curve = replay(clone_model(setup["model"]), res.events, res.parents,
                      HID, setup["layer"].eps, setup["val"],
                      checkpoints=[0.8, 0.6], hook=hook)
    assert len(seen) == 2 and seen[0] > seen[1]
    densities = [d for d, _ in curve]
    for d in seen:
        assert densities.count(d) == 2  # pre- and post-recovery points


def test_ft_block_recovers_pruning_damage(setup):
    model = clone_model(setup["model"])
    rng = np.random.default_rng(3)
    for _ in range(6):
        model = prune_unit(model, int(rng.integers(model.W1.shape[1])))
    before = model.loss(*setup["val"])
    ft_block(model, setup["train_ids"], steps=300, lr=1e-3, seed=2, batch=64)
    after = model.loss(*setup["val"])
    assert after < before


def test_encode_hook_with_recalibration(setup):
    def hook(model, layer, density, step):
        ft_block(model, setup["train_ids"], steps=30, lr=3e-4,
                 seed=4, batch=32)
        return model, calibrated_layer(model, setup["calib"])

    res = base_map(setup, checkpoints=[0.75], checkpoint_hook=hook)
    assert res.final_model.W1.shape[1] == res.events[-1].n_after
    densities = [d for d, _ in res.val_curve]
    assert densities.count(0.75) == 2


def test_prune_order_replay_matches_and_budget_log(setup):
    order = list(np.argsort(setup["layer"].capacities()))
    _, curve = replay_prune_order(clone_model(setup["model"]), order, HID,
                                  setup["val"], stop_density=0.5)
    assert curve[0][0] == 1.0 and curve[-1][0] <= 0.5
    log = FTLog()
    log.add("A", "empirical", 0.8, 500, 3e-4, 2.0, 1.9)
    log.add("A", "empirical", 0.6, 500, 3e-4, 2.1, 1.95)
    log.add("B", "empirical", 0.8, 500, 3e-4, 2.0, 1.9)
    assert log.total_steps("A", "empirical") == 1000
    assert log.total_steps("B", "empirical") == 500
