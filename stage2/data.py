"""character-level tiny Shakespeare"""

from __future__ import annotations
import collections
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "tinyshakespeare.txt")
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
              "master/data/tinyshakespeare/input.txt")


def load_corpus(path=CORPUS_PATH):
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(CORPUS_URL, path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class CharVocab:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        dtype = np.uint8 if len(self.chars) <= 256 else np.int64
        return np.fromiter((self.stoi[c] for c in text), dtype=dtype,
                           count=len(text))


def train_val_split(ids, val_fraction=0.1):
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:]


def sample_contexts(ids, k, n, rng):
    pos = rng.integers(0, len(ids) - k, size=n)
    ctx = ids[pos[:, None] + np.arange(k)[None, :]]
    tgt = ids[pos + k]
    return ctx, tgt


def all_contexts(ids, k, start, n):
    pos = start + np.arange(n)
    ctx = ids[pos[:, None] + np.arange(k)[None, :]]
    tgt = ids[pos + k]
    return ctx, tgt


def zipf_tables(text):
    char_counts = np.array(sorted(collections.Counter(text).values(),
                                  reverse=True), dtype=float)
    word_counts = np.array(sorted(collections.Counter(text.lower().split()).values(),
                                  reverse=True), dtype=float)
    return char_counts, word_counts
