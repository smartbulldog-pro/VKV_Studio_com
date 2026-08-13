"""
Local inference server for the persona stress-test — loads the fine-tuned Synapse
LoRA adapter (no GGUF needed) and serves POST /chat {messages:[...]} -> {reply}.
Uses the canonical v2.1 system prompt (read from the dataset) + prod-realistic sampling.
Run:  training/.venv-cuda/Scripts/python.exe training/eval/test_server.py [--adapter DIR] [--port 8777]
"""
import os, json, argparse, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from unsloth import FastModel
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=r"c:/projects/VKVstudio/training/output/synapse-v2-e2b-lora")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--seq", type=int, default=3072)
    ap.add_argument("--maxnew", type=int, default=2048)
    args = ap.parse_args()

    DS = r"c:/projects/VKVstudio/.system/datasets/synapse_v2.jsonl"
    with open(DS, encoding="utf-8") as f:
        first = json.loads(f.readline())
    SYS = next(m["content"] for m in first["messages"] if m["role"] == "system")

    print(f"Loading adapter: {args.adapter}", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=args.adapter, max_seq_length=args.seq, dtype=None,
        load_in_4bit=True, local_files_only=True,
    )
    FastModel.for_inference(model)
    # Gemma-4 returns a multimodal Processor; its apply_chat_template wants content
    # BLOCKS. Training rendered plain-string content via the underlying tokenizer,
    # so use that here to match exactly.
    tokenizer = getattr(processor, "tokenizer", processor)

    lock = __import__("threading").Lock()

    def generate(messages):
        msgs = [{"role": "system", "content": SYS}] + messages
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        attn = torch.ones_like(ids)
        with lock:
            out = model.generate(
                input_ids=ids, attention_mask=attn, max_new_tokens=args.maxnew,
                temperature=0.3, top_p=0.9, top_k=30,
                repetition_penalty=1.15, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            reply = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
            del out, ids, attn
            torch.cuda.empty_cache()
        return reply

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"ok")
        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                reply = generate(body.get("messages", []))
                data = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err))); self.end_headers(); self.wfile.write(err)

    print(f"MODEL LOADED — serving on http://127.0.0.1:{args.port}  (POST /chat)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
