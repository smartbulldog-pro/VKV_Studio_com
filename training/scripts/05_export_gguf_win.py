"""
============================================================
Шаг 5: Экспорт LoRA → Q8_0 GGUF (Windows), Synapse v2.3
============================================================
VERIFIED path (no cmake, no separate quantize):
  1. unsloth merge adapter → bf16 HF dir  (save_pretrained_merged, GPU)
  2. llama.cpp convert_hf_to_gguf.py --outtype q8_0  (pure Python, CPU)
     — the unsloth-bundled checkout at ~/.unsloth/llama.cpp FULLY supports the
     `gemma4` arch (conversion/gemma.py: @ModelBase.register("Gemma4ForConditionalGeneration")).
Использование: python scripts/05_export_gguf_win.py --model E2B [--quant q8_0]
============================================================
"""
import os, argparse, subprocess, sys, glob, shutil
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["E2B", "E4B"], default="E2B")
    ap.add_argument("--quant", default="q8_0")
    args = ap.parse_args()
    TD = "c:/projects/VKVstudio/training"
    LORA = os.path.join(TD, "output", f"synapse-v2-{args.model.lower()}-lora")
    MERGED = os.path.join(TD, "output", f"_merged_{args.model.lower()}")
    OUT_GGUF = os.path.join(TD, f"synapse-v2-{args.model.lower()}-{args.quant}.gguf")
    LLAMA = "c:/Users/user/.unsloth/llama.cpp"
    CONV = os.path.join(LLAMA, "convert_hf_to_gguf.py")
    PY = os.path.join(TD, ".venv-cuda", "Scripts", "python.exe")

    print(f"╔═ Шаг 5: {args.model} LoRA → {args.quant.upper()} GGUF ═╗")
    assert os.path.exists(f"{LORA}/adapter_model.safetensors"), f"no adapter at {LORA}"
    assert os.path.exists(CONV), f"no converter at {CONV}"

    # ── 1. merge adapter → bf16 HF ───────────────────────────────
    print("📦 [1/3] Мёрдж адаптера в bf16 (merged_16bit)...")
    from unsloth import FastModel
    model, tok = FastModel.from_pretrained(
        model_name=LORA, max_seq_length=2048, dtype=None,
        load_in_4bit=False, local_files_only=True,
    )
    if os.path.exists(MERGED): shutil.rmtree(MERGED)
    model.save_pretrained_merged(MERGED, tok, save_method="merged_16bit")
    del model
    import torch; torch.cuda.empty_cache()
    print(f"   ✅ merged → {MERGED}")

    # ── 2. convert_hf_to_gguf.py --outtype q8_0 ──────────────────
    print(f"🔄 [2/3] Конвертация → {args.quant} GGUF (gemma4, no cmake)...")
    if os.path.exists(OUT_GGUF): os.remove(OUT_GGUF)
    # run from the llama.cpp dir so the `conversion` pkg imports; use bundled gguf-py first on path
    env = dict(os.environ, PYTHONPATH=os.path.join(LLAMA, "gguf-py"))
    r = subprocess.run([PY, CONV, MERGED, "--outfile", OUT_GGUF, "--outtype", args.quant],
                       cwd=LLAMA, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("   ⚠️ convert stderr tail:\n" + "\n".join(r.stderr.splitlines()[-25:]))
        sys.exit(1)
    print("   ✅ converted")

    # ── 3. verify + cleanup ──────────────────────────────────────
    assert os.path.exists(OUT_GGUF), "GGUF not produced"
    size_gb = os.path.getsize(OUT_GGUF) / 1024**3
    print(f"💾 [3/3] {OUT_GGUF}  ({size_gb:.2f} GB)")
    # sanity: read arch via gguf
    try:
        from gguf import GGUFReader
        rd = GGUFReader(OUT_GGUF)
        f = rd.fields.get("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8", "replace") if f else "?"
        print(f"   arch={arch} (expect gemma4)")
    except Exception as e:
        print(f"   (arch check skipped: {e})")
    shutil.rmtree(MERGED, ignore_errors=True)   # free ~10-16GB intermediate
    print("═" * 50)
    print("  ✅ ГОТОВО. NEXT (mandatory): python training/scripts/bake_eos.py  →")
    print("     inference/models/synapse-<model>-q8-eosfix-v<ver>.gguf (bakes EOS/EOT id 106, Unsloth #5386).")
    print("  ⚠️ Prod swap is an ENV FLIP, not a file copy: point SYNAPSE_ROUTER_E2B_FILE /")
    print("     SYNAPSE_ROUTER_E4B_FILE at the eosfix gguf AND set SYNAPSE_SYSTEM_PROMPT_FILE=")
    print("     inference/synapse_v24_system_prompt.txt (the model's MATCHED training prompt).")
    print("     Move the model files and the prompt file TOGETHER — a train/serve prompt")
    print("     mismatch re-creates the degeneration this pipeline exists to prevent.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
