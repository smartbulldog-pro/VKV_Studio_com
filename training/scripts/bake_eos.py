"""Bake eos=106 + eot=106 into the freshly-exported Q8 GGUFs (Unsloth #5386 turn-end fix).
Writes to NEW *-eosfix-v24.gguf names in inference/models/ — does NOT overwrite the prior
baseline eosfix GGUFs, and NEVER touches the live prod synapse-q8.gguf."""
import subprocess, sys, os
from pathlib import Path

# Every path below used to be an absolute literal off this machine's C:\ drive,
# including one under the author's home directory. That made the script
# unrunnable by anyone else and leaked a local layout into a public repo for no
# benefit. Locations are now derived from where this file sits, and each can be
# overridden by an environment variable for a checkout laid out differently.
REPO = Path(__file__).resolve().parents[2]
TD = Path(os.getenv("VKV_TRAINING_DIR") or REPO / "training")
MODELS = Path(os.getenv("VKV_MODELS_DIR") or REPO / "inference" / "models")
# llama.cpp is a sibling checkout, not part of this repo. Unsloth clones it to
# ~/.unsloth by default; set VKV_LLAMA_CPP if yours lives elsewhere.
LLAMA = Path(os.getenv("VKV_LLAMA_CPP") or Path.home() / ".unsloth" / "llama.cpp")
# The CUDA training venv, falling back to whatever interpreter is running this
# — the gguf scripts need no CUDA, so a plain interpreter does the job.
_venv_py = TD / ".venv-cuda" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PY = str(_venv_py if _venv_py.exists() else sys.executable)

JOBS = [
    ("E4B", TD / "synapse-v2-e4b-q8_0.gguf", MODELS / "synapse-e4b-q8-eosfix-v24.gguf"),
    ("E2B", TD / "synapse-v2-e2b-q8_0.gguf", MODELS / "synapse-e2b-q8-eosfix-v24.gguf"),
]
assert LLAMA.exists(), f"llama.cpp checkout not found at {LLAMA} — set VKV_LLAMA_CPP"
env = dict(os.environ, PYTHONPATH=str(LLAMA / "gguf-py"))
env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"

for variant, src, dst in JOBS:
    assert src.exists(), f"missing exported GGUF: {src}"
    print(f"[{variant}] bake eos=106/eot=106 : {src.name} -> {dst.name}", flush=True)
    if dst.exists():
        dst.unlink()
    r = subprocess.run(
        [PY, "-m", "gguf.scripts.gguf_new_metadata",
         "--special-token-by-id", "eos", "106",
         "--special-token-by-id", "eot", "106",
         str(src), str(dst)],
        cwd=str(LLAMA), env=env, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        print("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-20:]), flush=True)
        sys.exit(1)
    print(f"   OK -> {dst}  ({dst.stat().st_size/1024**3:.2f} GB)", flush=True)

print("BOTH EOS-BAKED", flush=True)
