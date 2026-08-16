"""
rag.retriever.Retriever.retrieve — the grounding-selection correctness path.
===========================================================================
retrieve() decides WHICH corpus chunks get injected as grounding, and getting it
wrong is exactly the failure RAG exists to prevent: score too loosely and the
model is grounded on an irrelevant fact (confident confabulation); filter by the
wrong language and a Russian question is answered from English facts. It had no
direct test. These lock its four gates with a deterministic stub embedder, so
cosine (== dot on the L2-normalized vectors the real embedder returns) is fully
controllable:

  * same-language filter — a query only ever scores chunks of its own language;
  * min_score floor — chunks below the soft floor are dropped (no grounding on
    weak matches), and an all-weak query grounds on nothing;
  * top_k cap — applied BEFORE the floor, so it bounds the candidate set;
  * rel_margin gap filter — weak fill-ins far below the best match are dropped.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_rag_retrieve.py -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import Retriever  # noqa: E402

# L2-normalized vectors, so _cosine (a dot product) equals cosine similarity.
# beta = [0.9, sqrt(1-0.81), 0] → unit length, cosine 0.9 against [1,0,0].
_BETA_Y = (1.0 - 0.81) ** 0.5  # 0.43589...
_VECTORS = {
    # corpus chunk texts (fact.<lang> values)
    "alpha en": [1.0, 0.0, 0.0],
    "alpha ru": [1.0, 0.0, 0.0],
    "beta en": [0.9, _BETA_Y, 0.0],
    "beta ru": [0.9, _BETA_Y, 0.0],
    "gamma en": [0.0, 1.0, 0.0],
    "gamma ru": [0.0, 0.0, 1.0],
    # query strings
    "find alpha": [1.0, 0.0, 0.0],  # 1.0 / 0.9 / 0.0 against alpha / beta / gamma
    "void query": [0.0, 0.0, 1.0],  # orthogonal to every EN chunk → all scores 0
}

# Dedicated fixture for the ground() char-cap assembly: three EN chunks with
# controlled TEXT LENGTHS and near-identical scores (all within rel_margin of the
# top, so all three survive retrieve()), plus an off-topic one and two queries.
_GBIG = "B" * 15
_GMID = "M" * 30
_GSML = "S" * 4
_VECTORS.update({
    _GBIG: [1.0, 0.0, 0.0],  # score 1.0
    _GMID: [0.98, (1.0 - 0.98 ** 2) ** 0.5, 0.0],  # score 0.98
    _GSML: [0.96, (1.0 - 0.96 ** 2) ** 0.5, 0.0],  # score 0.96
    "OFF": [0.0, 1.0, 0.0],  # score 0.0 → dropped by the floor
    "gq": [1.0, 0.0, 0.0],
    "gvoid": [0.0, 0.0, 1.0],  # orthogonal to every ground chunk
})


class _StubEmbedder:
    """Deterministic embed_fn: known text -> fixed unit vector (KeyError if a text
    was never mapped, which surfaces a broken test rather than a silent zero)."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(self, texts, kind):  # noqa: ANN001
        self.calls.append((len(texts), kind))
        return [list(_VECTORS[t]) for t in texts]


class _RetrieverTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        d = Path(self._dir.name)
        self.corpus_path = d / "corpus.json"
        self.cache_path = d / "corpus_vectors.json"  # absent → forces a fresh embed
        self.corpus_path.write_text(
            json.dumps(
                {
                    "facts": [
                        {"id": "a", "en": "alpha en", "ru": "alpha ru", "source": "s"},
                        {"id": "b", "en": "beta en", "ru": "beta ru", "source": "s"},
                        {"id": "c", "en": "gamma en", "ru": "gamma ru", "source": "s"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.embedder = _StubEmbedder()

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _make(self, **kw) -> Retriever:
        opts = dict(top_k=3, min_score=0.35, rel_margin=0.07)
        opts.update(kw)
        return Retriever(self.embedder, self.corpus_path, self.cache_path, **opts)

    @staticmethod
    def _keys(hits) -> list:
        return [h["key"] for h in hits]


class TestRetrieveGates(_RetrieverTestBase):
    def test_min_score_and_rel_margin_keep_only_the_strong_top(self) -> None:
        # scores en: alpha 1.0, beta 0.9, gamma 0.0.
        # min_score 0.35 drops gamma; rel_margin 0.07 → cutoff 0.93 drops beta (0.9).
        r = self._make()
        hits = r.retrieve("find alpha", lang="en")
        self.assertEqual(self._keys(hits), ["a.en"])
        self.assertAlmostEqual(hits[0]["score"], 1.0)

    def test_same_language_filter_never_crosses_locales(self) -> None:
        # Identical query, but lang='ru' must score ONLY the ru chunks.
        r = self._make()
        hits = r.retrieve("find alpha", lang="ru")
        self.assertEqual(self._keys(hits), ["a.ru"])
        self.assertTrue(all(h["lang"] == "ru" for h in hits))

    def test_nothing_above_the_floor_grounds_on_nothing(self) -> None:
        # 'void query' is orthogonal to every EN chunk → all scores 0 < min_score.
        r = self._make()
        self.assertEqual(r.retrieve("void query", lang="en"), [])

    def test_top_k_caps_candidates_before_the_floor(self) -> None:
        # Floor off, margin wide → only top_k gates the result. 3 en chunks score
        # 1.0/0.9/0.0; top_k=2 keeps the best two, so gamma can't appear even at 0.0.
        r = self._make(top_k=2, min_score=0.0, rel_margin=1.0)
        hits = r.retrieve("find alpha", lang="en")
        self.assertEqual(len(hits), 2)
        self.assertEqual(self._keys(hits), ["a.en", "b.en"])

    def test_lang_defaults_to_detected_when_not_supplied(self) -> None:
        # No explicit lang → detect_lang('find alpha') is 'en' → en chunks scored.
        r = self._make()
        hits = r.retrieve("find alpha")
        self.assertTrue(hits and all(h["lang"] == "en" for h in hits))

    def test_query_is_embedded_as_a_query_not_a_document(self) -> None:
        # The corpus embeds once as 'document'; the query embeds as 'query'. The
        # asymmetric-embedding contract matters for retrieval quality.
        r = self._make()
        r.retrieve("find alpha", lang="en")
        kinds = [kind for _, kind in self.embedder.calls]
        self.assertIn("document", kinds)
        self.assertEqual(kinds[-1], "query")


class TestGroundAssembly(unittest.TestCase):
    """ground() assembles the retrieved chunks into a capped context and must
    return sources that match EXACTLY what it injected."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        d = Path(self._dir.name)
        self.corpus_path = d / "corpus.json"
        self.cache_path = d / "corpus_vectors.json"
        # EN-only facts (no `ru` → no ru chunks); scores 1.0 / 0.98 / 0.96 / 0.0.
        self.corpus_path.write_text(
            json.dumps(
                {
                    "facts": [
                        {"id": "big", "en": _GBIG, "source": "sb"},
                        {"id": "mid", "en": _GMID, "source": "sm"},
                        {"id": "sml", "en": _GSML, "source": "ss"},
                        {"id": "off", "en": "OFF", "source": "so"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.embedder = _StubEmbedder()

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _make(self, **kw) -> Retriever:
        opts = dict(top_k=3, min_score=0.35, rel_margin=0.07, max_context_chars=1400)
        opts.update(kw)
        return Retriever(self.embedder, self.corpus_path, self.cache_path, **opts)

    def test_no_hits_is_a_no_op_so_the_caller_sends_the_raw_query(self) -> None:
        wrapped, sources = self._make().ground("gvoid")
        self.assertIsNone(wrapped)
        self.assertEqual(sources, [])

    def test_full_fit_injects_all_and_sources_match_and_query_is_wrapped(self) -> None:
        wrapped, sources = self._make().ground("gq")  # cap 1400: all three fit
        self.assertEqual([s["key"] for s in sources], ["big.en", "mid.en", "sml.en"])
        self.assertIsNotNone(wrapped)
        self.assertIn("gq", wrapped)  # the query is inside the wrapper
        for txt in (_GBIG, _GMID, _GSML):
            self.assertIn(txt, wrapped)
        # sources carry the rounded score + source tag, matching the injected chunks
        self.assertEqual(sources[0]["source"], "sb")

    def test_oversized_first_chunk_is_truncated_not_dropped(self) -> None:
        # cap 10: the top chunk (15 chars) alone exceeds it → hard-capped with an
        # ellipsis, still included; the rest can't fit.
        wrapped, sources = self._make(max_context_chars=10).ground("gq")
        self.assertEqual([s["key"] for s in sources], ["big.en"])
        self.assertIn("…", wrapped)

    def test_a_chunk_that_does_not_fit_is_skipped_but_a_smaller_later_one_still_gets_in(self) -> None:
        # cap 25: big (15) fits; mid (30) does not; sml (4) still fits after big.
        # Locks the "skip this one, try the next smaller" branch AND that sources
        # reflect exactly big+sml (never the skipped mid).
        wrapped, sources = self._make(max_context_chars=25).ground("gq")
        keys = [s["key"] for s in sources]
        self.assertEqual(keys, ["big.en", "sml.en"])
        self.assertNotIn("mid.en", keys)
        self.assertIn(_GSML, wrapped)
        self.assertNotIn(_GMID, wrapped)


if __name__ == "__main__":
    unittest.main()
