"""Run the Synapse battery against an ollama model. Each item = FRESH conversation.
Saves one JSON per item as it completes (crash-resumable: existing files skipped).
System prompt is baked into the ollama model (Modelfile SYSTEM)."""
import argparse, json, os, time, urllib.request

def chat(model, messages, maxnew, host="127.0.0.1:11434"):
    data = json.dumps({"model": model, "stream": False, "messages": messages,
                       "options": {"num_predict": maxnew, "temperature": 0.7, "top_p": 0.9}}).encode("utf-8")
    req = urllib.request.Request(f"http://{host}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["message"]["content"], r.get("eval_count"), r.get("done_reason")

def run_item(model, turns, maxnew):
    messages, exchange = [], []
    for u in turns:
        messages.append({"role": "user", "content": u})
        a, ntok, done = chat(model, messages, maxnew)
        messages.append({"role": "assistant", "content": a})
        exchange.append({"user": u, "assistant": a, "out_tokens": ntok, "done_reason": done})
    return exchange

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxnew", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = [json.loads(l) for l in open(args.battery, encoding="utf-8") if l.strip()]
    done = 0
    for it in items:
        fp = os.path.join(args.out, f"{it['id']}.json")
        if os.path.exists(fp):
            done += 1; continue
        t0 = time.time()
        try:
            ex = run_item(args.model, it["turns"], args.maxnew)
        except Exception as e:
            print(f"[ERR] {it['id']}: {str(e)[:120]}", flush=True); continue
        rec = {"id": it["id"], "tag": it["tag"], "lang": it["lang"],
               "note": it.get("note", ""), "turns": it["turns"], "exchange": ex,
               "gen_seconds": round(time.time() - t0, 1)}
        json.dump(rec, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        done += 1
        last = ex[-1]["assistant"][:80].replace("\n", " ")
        print(f"[{done}/{len(items)}] {it['id']} ({rec['gen_seconds']}s) {last}...", flush=True)
    print(f"[DONE] {done}/{len(items)} -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
