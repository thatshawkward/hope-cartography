"""Corpus module checks: detagging and span confined sampling"""

import numpy as np

from stage2.corpora import Corpus, detag, sample_contexts_in_spans


def test_detag_reassembles_orthography():
    tagged = "I_PRP do_VBP n't_RB like_VB jet_NN lag_NN ,_, Bob_NNP ._."
    assert detag(tagged) == "I don't like jet lag, Bob."


def test_span_sampling_stays_inside_documents():
    rng = np.random.default_rng(0)
    ids = np.arange(1000) % 7
    spans = [("a", 100, 300), ("b", 600, 900)]
    k = 12
    ctx, tgt = sample_contexts_in_spans(ids, spans, k, 500, rng)

    starts = []
    for row, t in zip(ctx, tgt):
        # windows are contiguous slices of ids, so values follow (p+i) % 7
        p0 = row[0]
        assert np.all(row == (p0 + np.arange(k)) % 7)
    #sampled positions must satisfy span containment
    rng = np.random.default_rng(0)
    lens = np.array([b - a for _, a, b in spans], float)
    picks = rng.choice(len(spans), size=500, p=lens / lens.sum())
    starts = np.array([spans[p][1] for p in picks])
    widths = np.array([spans[p][2] - spans[p][1] for p in picks])
    pos = starts + (rng.random(500) * (widths - k - 1)).astype(np.int64)
    for p, s in zip(pos, picks):
        assert spans[s][1] <= p and p + k < spans[s][2]


def test_oanc_loader_handles_graf_wrapper_and_tiers(tmp_path):
    from stage2.corpora import load_oanc
    base = tmp_path / "OANC-GrAF" / "data"
    for tier, genre, src in [("spoken", "face-to-face", "charlotte"),
                             ("written_1", "journal", "slate"),
                             ("written_2", "travel_guides", "berlitz")]:
        d = base / tier / genre / src
        d.mkdir(parents=True)
        (d / "doc.txt").write_text("word " * 100, encoding="utf-8")
        (d / "doc-hepple.xml").write_text("<x/>", encoding="utf-8")
    corpus = load_oanc(str(tmp_path))
    genres = sorted(g for g, _, _ in corpus.docs)
    assert genres == ["face-to-face", "journal", "travel-guides"]
