"""A Bengio style character MLP language model in pure NumPy"""

from __future__ import annotations
import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CharMLP:
    def __init__(self, vocab, k=8, d=16, hidden=256, seed=0):
        rng = np.random.default_rng(seed)
        self.k, self.d, self.h, self.v = k, d, hidden, vocab
        self.E = rng.normal(scale=0.1, size=(vocab, d))
        self.W1 = rng.normal(scale=np.sqrt(2.0 / (k * d)), size=(k * d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=np.sqrt(1.0 / hidden), size=(hidden, vocab))
        self.b2 = np.zeros(vocab)
        self._params = ("E", "W1", "b1", "W2", "b2")
        self._adam = {p: [np.zeros_like(getattr(self, p)),
                          np.zeros_like(getattr(self, p))] for p in self._params}
        self._t = 0

    def features(self, ctx):
        return self.E[ctx].reshape(ctx.shape[0], -1)

    def preactivations(self, ctx, chunk=65536):
        outs = []
        for s in range(0, ctx.shape[0], chunk):
            x = self.features(ctx[s:s + chunk])
            outs.append(x @ self.W1 + self.b1)
        return np.concatenate(outs, axis=0)

    def loss(self, ctx, tgt):
        x = self.features(ctx)
        h = np.maximum(x @ self.W1 + self.b1, 0.0)
        p = _softmax(h @ self.W2 + self.b2)
        return float(-np.log(p[np.arange(len(tgt)), tgt] + 1e-12).mean())

    def loss_and_grads(self, ctx, tgt):
        B = ctx.shape[0]
        x = self.features(ctx)
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ self.W2 + self.b2
        p = _softmax(logits)
        loss = float(-np.log(p[np.arange(B), tgt] + 1e-12).mean())

        dlogits = p.copy()
        dlogits[np.arange(B), tgt] -= 1.0
        dlogits /= B
        grads = {}
        grads["W2"] = h.T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (h_pre > 0.0)
        grads["W1"] = x.T @ dh_pre
        grads["b1"] = dh_pre.sum(axis=0)
        dx = (dh_pre @ self.W1.T).reshape(B, self.k, self.d)
        dE = np.zeros_like(self.E)
        np.add.at(dE, ctx.reshape(-1), dx.reshape(-1, self.d))
        grads["E"] = dE
        return loss, grads

    def adam_step(self, grads, lr=1.5e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for p in self._params:
            g = grads[p]
            m, v = self._adam[p]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g * g
            mhat = m / (1 - beta1**self._t)
            vhat = v / (1 - beta2**self._t)
            getattr(self, p)[...] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, train_ids, steps=25000, batch=256, lr=1.5e-3, seed=1,
            log_every=5000, val=None):
        rng = np.random.default_rng(seed)
        from .data import sample_contexts
        history = []
        for step in range(1, steps + 1):
            ctx, tgt = sample_contexts(train_ids, self.k, batch, rng)
            loss, grads = self.loss_and_grads(ctx, tgt)
            self.adam_step(grads, lr=lr)
            if step % log_every == 0 or step == 1:
                msg = f"step {step:>6}  train batch loss {loss:.4f}"
                if val is not None:
                    msg += f"  val loss {self.loss(*val):.4f}"
                history.append(msg)
                print(msg, flush=True)
        return history


def gradient_check(seed=3, tol=2e-5):
    rng = np.random.default_rng(seed)
    model = CharMLP(vocab=11, k=3, d=4, hidden=7, seed=seed)
    ctx = rng.integers(0, 11, size=(5, 3))
    tgt = rng.integers(0, 11, size=5)
    _, grads = model.loss_and_grads(ctx, tgt)
    eps = 1e-6
    worst = 0.0
    for p in model._params:
        arr = getattr(model, p)
        flat = arr.reshape(-1)
        for idx in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps
            lp = model.loss(ctx, tgt)
            flat[idx] = old - eps
            lm = model.loss(ctx, tgt)
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[p].reshape(-1)[idx]
            worst = max(worst, abs(num - ana) / max(1.0, abs(num), abs(ana)))
    return worst < tol, worst
