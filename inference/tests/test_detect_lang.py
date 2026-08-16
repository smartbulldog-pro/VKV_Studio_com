"""
rag.retriever.detect_lang — the RU/EN routing heuristic.
=======================================================
detect_lang decides which corpus and which system instructions a query is
answered under, so a wrong verdict answers a Russian visitor from the English
corpus with English instructions (and vice-versa). The function had no direct
test; this locks its documented behaviour, especially the one regression it was
written to fix — the ratio is taken over LETTERS only, not the whole string, so
the dots/spaces/caps of a Latin product name inside a Russian sentence no longer
vote for English and push a genuinely Russian question under the threshold.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_detect_lang.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import detect_lang  # noqa: E402


class TestDetectLang(unittest.TestCase):
    def test_empty_and_letterless_default_to_en(self) -> None:
        # No letters to weigh → the safe default is English.
        for text in ("", "   ", "12345", "!!! ... ???", "42 + 7 = 49"):
            self.assertEqual(detect_lang(text), "en", text)

    def test_plain_english(self) -> None:
        for text in ("Hello world", "What model do you run?", "vLLM", "GPT-4 is great"):
            self.assertEqual(detect_lang(text), "en", text)

    def test_plain_russian(self) -> None:
        for text in ("Привет, мир!", "Какую модель ты используешь?", "Расскажи о себе"):
            self.assertEqual(detect_lang(text), "ru", text)

    def test_russian_sentence_with_latin_product_names_is_ru(self) -> None:
        # The documented regression. Naming things in Latin is normal in a Russian
        # technical sentence, and this site is about products with Latin names.
        for text in (
            "Ты используешь llama.cpp или vLLM?",
            "Это React?",  # 3 Cyrillic / 5 Latin letters = 0.375 > 0.30
            "Развёрнут на Cloudflare Pages?",
        ):
            self.assertEqual(detect_lang(text), "ru", text)

    def test_mostly_english_with_a_little_cyrillic_stays_en(self) -> None:
        # cyr 6 / (6 + 15 latin) = 0.285 < 0.30 → English wins, as intended.
        self.assertEqual(detect_lang("привет abcdefghijklmno"), "en")

    def test_threshold_counts_letters_only_not_punctuation(self) -> None:
        # "Это React?" is 3/8 = 0.375 counting letters (→ ru), but only 3/10 = 0.30
        # counting the whole string (→ en, the old bug). If this ever flips to "en"
        # again, the letters-only fix has regressed.
        self.assertEqual(detect_lang("Это React?"), "ru")

    def test_return_value_is_only_ever_a_known_lang_code(self) -> None:
        for text in ("", "mixed текст 123", "?!.", "Ελληνικά", "日本語のテスト"):
            self.assertIn(detect_lang(text), ("en", "ru"), text)


if __name__ == "__main__":
    unittest.main()
