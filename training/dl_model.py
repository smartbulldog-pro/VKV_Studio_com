"""Resumable base-model download. Usage: python training/dl_model.py E2B   (or E4B)
Classic HF downloader (Xet OFF = real resume of the .incomplete on every retry),
30s read-timeout so a stuck connection raises and the loop resumes — no restart-from-zero."""
import sys, os, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download

M = (sys.argv[1] if len(sys.argv) > 1 else "E2B").upper()
repo = f"google/gemma-4-{M}-it"
dest = f"C:/projects/VKVstudio/training/models/gemma-4-{M}-it"
PATS = ["*.safetensors", "*.safetensors.index.json", "config.json",
        "generation_config.json", "tokenizer*", "*.model",
        "chat_template.jinja", "processor_config.json", "special_tokens_map.json"]

print(f"Downloading {repo}  ->  {dest}", flush=True)
for attempt in range(1, 400):
    try:
        print(f"\n[attempt {attempt}] connecting...", flush=True)
        snapshot_download(repo, local_dir=dest, allow_patterns=PATS, max_workers=4)
        print(f"\n✅ DONE — {repo} fully downloaded.", flush=True)
        break
    except Exception as e:
        print(f"[attempt {attempt}] dropped: {type(e).__name__}: {str(e)[:150]} — resuming in 5s", flush=True)
        time.sleep(5)
else:
    print("❌ EXHAUSTED retries", flush=True)
    sys.exit(1)
