"""
Unit tests for the pure-Python WER metric (eval/wer.py).
========================================================
Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_wer -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.wer import corpus_wer, normalize_text, word_error_rate  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_lowercases_and_strips_punct(self) -> None:
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  a   b\tc\n"), "a b c")

    def test_keeps_cyrillic(self) -> None:
        self.assertEqual(normalize_text("Привет, мир."), "привет мир")


class TestWer(unittest.TestCase):
    def test_perfect_match(self) -> None:
        r = word_error_rate("the quick brown fox", "the quick brown fox")
        self.assertEqual(r.wer, 0.0)
        self.assertEqual(r.hits, 4)

    def test_only_punctuation_and_case_differ(self) -> None:
        r = word_error_rate("Hello world", "hello, WORLD!")
        self.assertEqual(r.wer, 0.0)

    def test_one_substitution(self) -> None:
        r = word_error_rate("the quick brown fox", "the quick red fox")
        self.assertEqual(r.substitutions, 1)
        self.assertEqual(r.deletions, 0)
        self.assertEqual(r.insertions, 0)
        self.assertAlmostEqual(r.wer, 1 / 4)

    def test_one_deletion(self) -> None:
        r = word_error_rate("a b c d", "a b d")
        self.assertEqual(r.deletions, 1)
        self.assertAlmostEqual(r.wer, 1 / 4)

    def test_one_insertion(self) -> None:
        r = word_error_rate("a b c", "a b x c")
        self.assertEqual(r.insertions, 1)
        self.assertAlmostEqual(r.wer, 1 / 3)

    def test_empty_reference_nonempty_hyp(self) -> None:
        r = word_error_rate("", "some words here")
        self.assertEqual(r.wer, 1.0)
        self.assertEqual(r.insertions, 3)

    def test_empty_both(self) -> None:
        self.assertEqual(word_error_rate("", "").wer, 0.0)

    def test_full_miss(self) -> None:
        r = word_error_rate("alpha beta", "gamma delta")
        self.assertEqual(r.wer, 1.0)

    def test_russian(self) -> None:
        r = word_error_rate("привет как дела", "привет как дела")
        self.assertEqual(r.wer, 0.0)
        r2 = word_error_rate("привет как дела", "привет как жизнь")
        self.assertAlmostEqual(r2.wer, 1 / 3)

    def test_as_pct(self) -> None:
        r = word_error_rate("a b c d", "a b c x")
        self.assertEqual(r.as_pct(), 25.0)


class TestCorpusWer(unittest.TestCase):
    def test_micro_average(self) -> None:
        pairs = [
            ("a b c d", "a b c d"),      # 0/4
            ("a b c d", "a b x y"),      # 2/4
        ]
        r = corpus_wer(pairs)
        # micro-average: total edits 2 over total ref words 8
        self.assertAlmostEqual(r.wer, 2 / 8)
        self.assertEqual(r.ref_words, 8)

    def test_empty_corpus(self) -> None:
        self.assertEqual(corpus_wer([]).wer, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
