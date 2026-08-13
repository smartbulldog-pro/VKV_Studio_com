# Fire the 22 gate queries at the LIVE /api/chat (RAG integration test).
# Pins E2B (model override) to match the gate's E2B-only comparison.
import json, subprocess, tempfile, os, sys
from pathlib import Path

SP = Path(__file__).parent
API = "http://localhost:8000/api/chat"
# battery file + output via argv (default: the 22-prompt gate set)
BATTERY = Path(sys.argv[1]) if len(sys.argv) > 1 else SP / "gate_baseline.jsonl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SP / "battery_results_chat_rag.json"

def post(message, tries=6):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps({"message": message, "history": [], "model": "e2b"}).encode()); os.close(fd)
    try:
        for _ in range(tries):
            # capture BYTES and decode UTF-8 explicitly — text=True uses the Windows
            # locale (cp1251) and mangles/crashes on the Cyrillic UTF-8 curl output.
            p = subprocess.run(["curl", "-s", "-m", "180", "-X", "POST", API,
                                "-H", "Content-Type: application/json", "--data-binary", "@"+path],
                               capture_output=True)
            out = p.stdout.decode("utf-8", errors="replace")
            try:
                d = json.loads(out)
                if "response" in d:
                    return d
            except Exception:
                pass
        return {"response": "", "model": None, "_err": out[:200]}
    finally:
        os.unlink(path)

rows = [json.loads(l) for l in open(BATTERY, encoding="utf-8") if l.strip()]
results = []
for i, row in enumerate(rows):
    d = post(row["turns"][0])
    results.append({"id": row["id"], "tag": row["tag"], "lang": row["lang"], "bucket": row.get("bucket", ""),
                    "turns": [{"text": d.get("response", ""), "model": d.get("model")}]})
    print(f"[{i+1:2}/{len(rows)}] {row['id']:<22} model={d.get('model')} chars={len(d.get('response') or '')}", flush=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
