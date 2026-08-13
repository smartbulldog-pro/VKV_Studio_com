"""Export E4B then E2B adapters to Q8_0 GGUF, sequentially, with a sentinel per model."""
import subprocess, sys, time, os
from pathlib import Path

TD = Path(r"C:\projects\VKVstudio\training")
PY = str(TD / ".venv-cuda" / "Scripts" / "python.exe")
EXPORT = str(TD / "scripts" / "05_export_gguf_win.py")

for variant in ["E4B", "E2B"]:
    t0 = time.time()
    print(f"\n===== EXPORT {variant} @ {time.strftime('%H:%M:%S')} =====", flush=True)
    logf = open(TD / f"export_{variant.lower()}.log", "w", encoding="utf-8", buffering=1)
    r = subprocess.run([PY, EXPORT, "--model", variant], cwd=str(TD),
                       stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    dt = int(time.time() - t0)
    gguf = TD / f"synapse-v2-{variant.lower()}-q8_0.gguf"
    if r.returncode == 0 and gguf.exists():
        (TD / f"EXPORT_OK_{variant}").write_text(
            f"{gguf} {gguf.stat().st_size} {dt}s\n", encoding="utf-8")
        print(f"  OK {variant} in {dt}s -> {gguf.name} "
              f"({gguf.stat().st_size/1024**3:.2f} GB)", flush=True)
    else:
        print(f"  FAIL {variant} rc={r.returncode} after {dt}s "
              f"(see export_{variant.lower()}.log)", flush=True)
        sys.exit(1)

print("\nBOTH EXPORTS DONE", flush=True)
