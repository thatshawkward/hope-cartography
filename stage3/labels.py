"""Linguistic labels for hidden units"""

from __future__ import annotations

import numpy as np

VOWELS = set("aeiouAEIOU")
PUNCT = set(".,:;!?'-")


def context_properties(ctx, tgt, vocab):
    chars = np.array(vocab.chars)
    last = chars[ctx[:, -1]]
    nxt = chars[tgt]

    def isin(arr, charset):
        return np.isin(arr, list(charset)).astype(float)

    props = {
        "last=space": (last == " ").astype(float),
        "last=newline": (last == "\n").astype(float),
        "last=vowel": isin(last, VOWELS),
        "last=punct": isin(last, PUNCT),
        "next=space": (nxt == " ").astype(float),
        "next=newline": (nxt == "\n").astype(float),
        "next=vowel": isin(nxt, VOWELS),
        "next=upper": np.char.isupper(nxt.astype(str)).astype(float),
    }
    return props


def label_units(h_pre, props, threshold=0.25):
    n, H = h_pre.shape
    hz = (h_pre - h_pre.mean(axis=0)) / (h_pre.std(axis=0) + 1e-12)
    names = list(props.keys())
    corr = np.zeros((len(names), H))
    for k, name in enumerate(names):
        p = props[name]
        pz = (p - p.mean()) / (p.std() + 1e-12)
        corr[k] = pz @ hz / n
    best = np.abs(corr).argmax(axis=0)
    strength = np.abs(corr)[best, np.arange(H)]
    labels = np.array([names[b] if s >= threshold else "distributed"
                       for b, s in zip(best, strength)])
    return labels, corr, names


def top_contexts(h_pre, ctx, vocab, unit, k=5):
    order = np.argsort(h_pre[:, unit])[::-1][:k]
    chars = np.array(vocab.chars)
    out = []
    for r in order:
        s = "".join(chars[ctx[r]])
        out.append(s.replace("\n", "\\n"))
    return out
