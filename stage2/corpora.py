from __future__ import annotations
import io
import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
import numpy as np

MASC_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
            "gh-pages/packages/corpora/masc_tagged.zip")

_ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}
_REPL = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
         "\t": " "}

#attach clitics and closing punctuation that tokenization split off
_DETOK = [
    (re.compile(r"\s+(n't|'re|'ve|'ll|'d|'m|'s|na|nt)\b"), r"\1"),
    (re.compile(r"\s+([.,;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$])\s+"), r"\1"),
    (re.compile(r"``\s*"), '"'),
    (re.compile(r"\s*''"), '"'),
    (re.compile(r" {2,}"), " "),
]


def clean_text(s):
    s = unicodedata.normalize("NFKC", s)
    for a, b in _REPL.items():
        s = s.replace(a, b)
    s = "".join(c if c in _ALLOWED else " " for c in s)
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def detag(tagged):
    words = []
    for tok in tagged.split():
        w = tok.rsplit("_", 1)[0] if "_" in tok else tok
        if w:
            words.append(w)
    s = " ".join(words)
    for pat, rep in _DETOK:
        s = pat.sub(rep, s)
    return s


@dataclass
class Corpus:
    name: str
    text: str                       
    docs: list = field(default_factory=list)   
    train_docs: list = field(default_factory=list)
    val_docs: list = field(default_factory=list)

    def split_spans(self, val_fraction=0.1, seed=0):
        rng = np.random.default_rng(seed)
        by_genre = {}
        for d in self.docs:
            by_genre.setdefault(d[0], []).append(d)
        train, val = [], []
        for g, ds in sorted(by_genre.items()):
            ds = list(ds)
            rng.shuffle(ds)
            n_val = max(1, int(round(len(ds) * val_fraction))) if len(ds) > 1 else 0
            val += ds[:n_val]
            train += ds[n_val:]
        self.train_docs, self.val_docs = train, val
        return train, val

    def genre_volumes(self, docs=None):
        vols = {}
        for g, a, b in (docs or self.docs):
            vols[g] = vols.get(g, 0) + (b - a)
        return dict(sorted(vols.items(), key=lambda kv: -kv[1]))


def _assemble(name, records):
    parts, docs, pos = [], [], 0
    for genre, doc in records:
        if len(doc) < 200:
            continue
        parts.append(doc)
        docs.append((genre, pos, pos + len(doc)))
        pos += len(doc) + 2                      # the "\n\n" joiner
    return Corpus(name=name, text="\n\n".join(parts), docs=docs)


def load_masc(cache_dir="data/masc", url=MASC_URL):
    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "masc_tagged.zip")
    if not os.path.exists(zpath):
        import urllib.request
        urllib.request.urlretrieve(url, zpath)
    zf = zipfile.ZipFile(zpath)
    cats = {}
    with zf.open("masc_tagged/categories.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            if line.strip():
                path, genre = line.split()
                cats[path] = genre
    records = []
    for path, genre in sorted(cats.items()):
        with zf.open(f"masc_tagged/{path}") as fh:
            raw = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
        records.append((genre, clean_text(detag(raw))))
    return _assemble("masc", records)


def load_oanc(root="data/oanc"):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"OANC not found at {root!r}. anc.org's TLS certificate lapsed in "
            "2025, so acquire the corpus one of these ways, then extract into "
            f"{root!r}:\n"
            "  1) plain HTTP (no certificate involved):\n"
            "     curl -L -o OANC_GrAF.zip http://www.anc.org/OANC/OANC_GrAF.zip\n"
            "  2) knowingly accept the expired certificate:\n"
            "     curl -kL -o OANC_GrAF.zip https://www.anc.org/OANC/OANC_GrAF.zip\n"
            "  3) no-bypass route via the Internet Archive:\n"
            "     https://web.archive.org/web/2024/http://www.anc.org/OANC/OANC_GrAF.zip\n"
            "Sanity: ~0.6 GB zip, ~7.4 GB / 62,090 files unpacked; the zip "
            "carries an OANC-GrAF/ wrapper directory, which is fine. The "
            "loader reads the plain .txt primaries and ignores annotation "
            "standoff.")
    records = []
    for dirpath, _, files in sorted(os.walk(root)):
        for fn in sorted(files):
            if not fn.endswith(".txt"):
                continue
            rel = os.path.relpath(dirpath, root)
            parts = [p for p in rel.split(os.sep) if p != "."]
            if "data" in parts:                
                parts = parts[parts.index("data") + 1:]
            genre = (parts[1] if len(parts) > 1 else
                     parts[0] if parts else "unknown").replace("_", "-")
            with open(os.path.join(dirpath, fn), encoding="utf-8",
                      errors="replace") as fh:
                records.append((genre, clean_text(fh.read())))
    if not records:
        raise FileNotFoundError(f"no .txt documents found under {root!r}")
    return _assemble("oanc", records)


def sample_contexts_in_spans(ids, spans, k, n, rng):
    lens = np.array([b - a for _, a, b in spans], dtype=float)
    ok = lens > k + 1
    spans = [s for s, o in zip(spans, ok) if o]
    lens = lens[ok]
    picks = rng.choice(len(spans), size=n, p=lens / lens.sum())
    starts = np.array([spans[p][1] for p in picks])
    widths = np.array([spans[p][2] - spans[p][1] for p in picks])
    pos = starts + (rng.random(n) * (widths - k - 1)).astype(np.int64)
    ctx = ids[pos[:, None] + np.arange(k)[None, :]]
    tgt = ids[pos + k]
    return ctx, tgt
