"""
STT WER harness — Gemma-4 native audio.
=======================================
Runs the Gemma STT backend over a set of reference clips and prints a
RU-WER / EN-WER table. (Whisper was removed 2026-07-06 when STT consolidated
onto Gemma-4 native audio, so this is now a single-backend accuracy check
rather than a Whisper-vs-Gemma gate.)

Design:
  * The Gemma backend currently raises NotImplementedError (audio-mmproj GGUF
    pending → main session). The harness catches that and reports the backend as
    "PENDING" rather than crashing, so the table and aggregation logic are proven
    and ready to light up the moment mmproj lands.

Usage (from inference/):
    # dry-run — just validate the manifest, load nothing:
    ./.venv/Scripts/python.exe -m eval.wer_harness --manifest eval/manifest.json --dry-run

    # once mmproj exists + llama-server --mmproj is up:
    ./.venv/Scripts/python.exe -m eval.wer_harness --manifest eval/manifest.json --backends gemma

Manifest format (JSON): {"clips": [{"file": "clips/ru_01.wav", "lang": "ru",
"reference": "эталонный текст"}, ...]}. Paths are relative to the manifest file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `import stt` / `from eval.wer import ...` when run from inference/.
_INFERENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_INFERENCE_DIR))

from eval.wer import corpus_wer, word_error_rate  # noqa: E402


def load_manifest(path: Path) -> "list[dict]":
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    base = path.parent
    for c in clips:
        c["_path"] = (base / c["file"]).resolve()
    return clips


def _make_backend(name: str):
    """Construct an STT backend by name. Import is local so --dry-run needs nothing.

    Explicit map (not get_stt()) so the harness stays independent of the
    SYNAPSE_STT_BACKEND env var.
    """
    import stt

    return {"gemma": stt.GemmaSTT}[name]()


def run_backend(name: str, clips: "list[dict]") -> "dict | None":
    """Transcribe every clip with one backend. Returns per-language results, or
    None if the backend is not available yet (e.g. Gemma mmproj pending)."""
    try:
        engine = _make_backend(name)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the table
        print(f"  [{name}] could not construct backend: {exc}")
        return None

    per_lang: dict[str, list[tuple[str, str]]] = {}
    pending = False
    for c in clips:
        path: Path = c["_path"]
        if not path.exists():
            print(f"  [{name}] MISSING clip: {path} — skipping")
            continue
        try:
            result = engine.transcribe(path.read_bytes())
            hyp = result.text
        except NotImplementedError as exc:
            pending = True
            print(f"  [{name}] PENDING — {exc}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] ERROR on {path.name}: {exc}")
            continue
        r = word_error_rate(c["reference"], hyp)
        per_lang.setdefault(c["lang"], []).append((c["reference"], hyp))
        print(f"  [{name}] {path.name} ({c['lang']}): WER={r.as_pct():.2f}%  hyp={hyp!r}")

    if pending:
        return None
    return {lang: corpus_wer(pairs) for lang, pairs in per_lang.items()}


def print_table(results: "dict[str, dict | None]") -> None:
    langs = ["ru", "en"]
    header = f"{'backend':<10} " + " ".join(f"{l.upper()+'-WER':>10}" for l in langs)
    print("\n" + header)
    print("-" * len(header))
    for name, res in results.items():
        if res is None:
            row = f"{name:<10} " + " ".join(f"{'PENDING':>10}" for _ in langs)
        else:
            cells = []
            for l in langs:
                cells.append(f"{res[l].as_pct():>9.2f}%" if l in res else f"{'-':>10}")
            row = f"{name:<10} " + " ".join(cells)
        print(row)
    print()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="STT WER harness (Gemma-4 native audio).")
    ap.add_argument("--manifest", default="eval/manifest.json", type=Path)
    ap.add_argument("--backends", nargs="+", default=[],
                    choices=["gemma"],
                    help="Backends to score. Empty + --dry-run just validates the manifest.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate manifest and list clips; load no models.")
    args = ap.parse_args(argv)

    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}\n"
              f"Copy eval/manifest.example.json → {manifest_path.name} and add clips "
              f"(see eval/README.md / eval/gen_clips.py).")
        return 2

    clips = load_manifest(manifest_path)
    n_ru = sum(1 for c in clips if c["lang"] == "ru")
    n_en = sum(1 for c in clips if c["lang"] == "en")
    print(f"Manifest: {len(clips)} clips ({n_ru} ru, {n_en} en)")
    missing = [str(c["_path"]) for c in clips if not c["_path"].exists()]
    if missing:
        print(f"  [!] {len(missing)} referenced clip(s) not on disk yet:")
        for m in missing:
            print(f"    - {m}")

    if args.dry_run or not args.backends:
        print("\n(dry-run: no backends loaded)" if not args.backends else "")
        return 0

    results = {name: run_backend(name, clips) for name in args.backends}
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
