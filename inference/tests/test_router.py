"""
router.route / is_complex — the per-turn E2B⇄E4B routing heuristic.
==================================================================
route() decides which fine-tune answers each message. It is a pure, dependency-
free heuristic the module itself advertises as "fully unit-testable with plain
strings" — but had no test. Misrouting matters: over-escalating sends cheap
conversational turns to the bigger model, and (with the router enabled)
under-escalating under-serves a genuinely hard ask. These lock every signal and
the enable/default wiring, with the thresholds monkeypatched so the test doesn't
drift when config defaults change.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_router.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import router  # noqa: E402


class _RouterBase(unittest.TestCase):
    _PATCHED = (
        "ROUTER_CHAR_THRESHOLD",
        "ROUTER_WORD_THRESHOLD",
        "ROUTER_ENABLED",
        "ROUTER_DEFAULT_VARIANT",
        "ROUTER_COMPLEX_KEYWORDS",
    )

    def setUp(self) -> None:
        self._saved = {n: getattr(router, n) for n in self._PATCHED}
        router.ROUTER_CHAR_THRESHOLD = 100
        router.ROUTER_WORD_THRESHOLD = 20
        router.ROUTER_ENABLED = True
        router.ROUTER_DEFAULT_VARIANT = "e2b"
        router.ROUTER_COMPLEX_KEYWORDS = ()  # empty → built-in bilingual keyword set

    def tearDown(self) -> None:
        for n, v in self._saved.items():
            setattr(router, n, v)


class TestIsComplex(_RouterBase):
    def test_empty_and_short_simple_are_not_complex(self) -> None:
        for msg in ("", "   ", "hi", "hey there", "thanks!"):
            self.assertFalse(router.is_complex(msg), msg)

    def test_bare_interrogatives_stay_simple(self) -> None:
        # Deliberate design: a short casual "why/how" is E2B territory.
        for msg in ("why?", "how are you", "почему?", "как дела"):
            self.assertFalse(router.is_complex(msg), msg)

    def test_length_signal(self) -> None:
        self.assertTrue(router.is_complex("x" * 100))  # exactly the char threshold
        self.assertFalse(router.is_complex("x" * 99))

    def test_word_count_signal_independent_of_length(self) -> None:
        twenty = " ".join("w" for _ in range(20))  # 20 words, 39 chars < 100
        self.assertLess(len(twenty), 100)
        self.assertTrue(router.is_complex(twenty))
        self.assertFalse(router.is_complex(" ".join("w" for _ in range(19))))

    def test_code_hints(self) -> None:
        for msg in ("```py\nx=1\n```", "def foo():", "class A:", "import os",
                    "a => b", "Foo::bar", "<div>hi</div>"):
            self.assertTrue(router.is_complex(msg), msg)

    def test_keywords_en_and_ru(self) -> None:
        for msg in ("please refactor this", "optimize the query", "step by step",
                    "отладь это", "спроектируй архитектуру", "по шагам"):
            self.assertTrue(router.is_complex(msg), msg)

    def test_keyword_is_case_insensitive(self) -> None:
        self.assertTrue(router.is_complex("REFACTOR this"))


class TestRoute(_RouterBase):
    def test_enabled_complex_goes_to_e4b(self) -> None:
        router.ROUTER_ENABLED = True
        self.assertEqual(router.route("please refactor this module"), router.VARIANT_E4B)

    def test_enabled_simple_goes_to_default(self) -> None:
        router.ROUTER_ENABLED = True
        self.assertEqual(router.route("hi there"), "e2b")

    def test_disabled_never_switches_even_for_complex(self) -> None:
        router.ROUTER_ENABLED = False
        # A clearly-complex prompt must still land on the default when routing is off.
        self.assertEqual(router.route("refactor this huge algorithm " + "x" * 200), "e2b")

    def test_route_only_ever_returns_a_valid_variant(self) -> None:
        for msg in ("", "hi", "refactor", "x" * 300, "отладь"):
            self.assertIn(router.route(msg), router.VALID_VARIANTS, msg)


class TestNormalizeVariant(_RouterBase):
    def test_valid_passthrough_and_case_whitespace(self) -> None:
        self.assertEqual(router._normalize_variant("e4b"), "e4b")
        self.assertEqual(router._normalize_variant("  E2B  "), "e2b")

    def test_garbage_and_empty_fall_back_to_e2b(self) -> None:
        for bad in ("", "e5b", "gpt", "e2b ; drop", None):
            self.assertEqual(router._normalize_variant(bad), "e2b", repr(bad))


if __name__ == "__main__":
    unittest.main()
