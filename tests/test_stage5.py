"""stage 5 repair mechanisms: re-projection exactness, SGD scoping, sync"""

import numpy as np
import pytest

from stage5.repair import block_output, reproject


class StubState:
    def __init__(self, X, W, b, Wo):
        self.X, self.W, self.b, self.Wo = X, W, b, Wo
        self.H = X @ W + b


def make_stub(rng, n_in=12, n=10, n_out=8, rows=64):
    X = rng.standard_normal((rows, n_in)).astype(np.float32)
    W = rng.standard_normal((n_in, n)).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    Wo = rng.standard_normal((n, n_out)).astype(np.float32)
    return StubState(X, W, b, Wo)


def test_reprojection_is_exact_on_full_basis():
    rng = np.random.default_rng(0)
    st = make_stub(rng)
    y = block_output(st)
    st.Wo = rng.standard_normal(st.Wo.shape).astype(np.float32)
    reproject(st, y)
    assert np.abs(block_output(st) - y).max() < 1e-4


def test_reprojection_recovers_dropped_duplicate():
    rng = np.random.default_rng(1)
    st = make_stub(rng)
    st.W[:, 1] = st.W[:, 0]
    st.b[1] = st.b[0]
    st.H = st.X @ st.W + st.b
    y = block_output(st)
    keep = list(range(1, st.W.shape[1]))
    st.W, st.b, st.Wo = st.W[:, keep], st.b[keep], st.Wo[keep]
    st.H = st.X @ st.W + st.b
    assert np.abs(block_output(st) - y).max() > 1e-3
    reproject(st, y)
    assert np.abs(block_output(st) - y).max() < 1e-4


@pytest.fixture
def tiny_gpt2():
    tf = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    cfg = tf.GPT2Config(n_layer=3, n_embd=32, n_head=2, n_positions=64,
                        vocab_size=97, resid_pdrop=0.0, embd_pdrop=0.0,
                        attn_pdrop=0.0)
    model = tf.GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_sgd_scope_block_touches_only_that_block(tiny_gpt2, monkeypatch):
    torch = pytest.importorskip("torch")
    import stage5.repair as R
    monkeypatch.setattr(R, "WINDOW", 16)
    model = tiny_gpt2
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    toks = np.random.default_rng(0).integers(0, 97, size=2000)
    R.sgd_recover(model, toks, steps=3, lr=1e-3, seed=0, scope="block",
                  block_ix=2)
    for k, v in model.named_parameters():
        changed = not torch.equal(before[k], v)
        assert changed == ("h.1.mlp" in k), k
    assert all(p.requires_grad for p in model.parameters())


def test_replay_with_reprojection_on_live_block(tiny_gpt2):
    torch = pytest.importorskip("torch")
    from gpt2map.run_gpt2_map import BlockState, blocks_of
    import stage5.repair as R
    model = tiny_gpt2
    rng = np.random.default_rng(2)
    X = rng.standard_normal((32, 32)).astype(np.float32)
    blk = blocks_of(model)[1].mlp
    H = X @ blk.c_fc.weight.detach().numpy() + blk.c_fc.bias.detach().numpy()
    st = BlockState(model, 2, H.mean(0), H.std(0) + 1e-12, X)
    y = R.block_output(st)
    h0 = st.n
    for u in list(rng.permutation(h0))[:h0 // 2]:
        st.drop([int(np.where(st.ids == u)[0][0])])
    R.reproject(st, y)
    st.push_to(model, 2)
    x = torch.from_numpy(rng.integers(0, 97, size=(1, 16)))
    assert np.isfinite(float(model(x, labels=x).loss))
    R.pull_from(st, model, 2)
    assert st.W.shape[1] == h0 - h0 // 2
    assert st.H.shape == (32, st.W.shape[1])
