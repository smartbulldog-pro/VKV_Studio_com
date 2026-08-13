# Near-miss experiment generator.
# Q: does WRONG-but-plausible retrieved context make E2B confabulate MORE than no context?
# Method: embed a small real-studio-fact seed corpus + the 22 gate queries via the live
# EmbeddingGemma (/api/embed), take same-language cosine TOP-1 with NO threshold, inject it.
# For fake-entity / out-of-corpus queries no chunk truly answers -> top-1 is a genuine near-miss.
# Emits: gate_nearmiss.jsonl (battery for run_battery.py) + nearmiss_scores.tsv (score separation).
import json, subprocess, tempfile, os
from pathlib import Path

SP = Path(__file__).parent
API = "http://localhost:8000/api/embed"

# --- Seed corpus: real, public studio facts only (NO age/health/location — deny-list). ---
# Bilingual; same-language retrieval is the design. Facts sourced from ground_truth_facts.md.
CORPUS = [
 # EN
 ("en","model","Synapse is Gemma 4 E2B (~2B parameters), fine-tuned with QLoRA using Unsloth, and served as a Q8_0 GGUF file."),
 ("en","runtime","Synapse runs server-side on the studio's GPU workstation via llama-server; in production it moves to an Oracle/Hetzner ARM host. It does not run in the user's browser."),
 ("en","router","The production router has two models: E2B 'junior' (~2B) for simple turns and E4B 'mid' (~8B) for complex ones; one is in VRAM at a time. There is no 70B version."),
 ("en","training","Synapse was fine-tuned with QLoRA (plus Unsloth) on Gemma 4 E2B. It is not a full fine-tune and did not use RLHF."),
 ("en","knowledge","Synapse has a knowledge cutoff around June 2026, no internet access, and no terminal or filesystem access. It does not train or learn from conversations."),
 ("en","studio","VKVstudio builds premium, high-performance vanilla-first websites and AI microservices. It is a web-development studio, not an AR/VR studio."),
 ("en","philosophy","The studio's engineering philosophy: Lighthouse 100 is the baseline not the goal; every kilobyte must be justified; vanilla-first, zero-trust, autonomy and fault tolerance."),
 ("en","team","VKVstudio is Valery's solo studio. There is no co-founder, no CTO, and no named team or roster on record."),
 ("en","career","Valery has about 20 years of experience, moving from product photographer to in-house marketer at industrial plants to web developer and studio founder to AI product creator."),
 ("en","tools","The Lab has three tools: a Tokenizer Profiler, a Prompt Architect, and an Embedding Explorer. The Embedding Explorer computes vectors server-side via Google EmbeddingGemma; it is backend-only with no public API."),
 ("en","boundary","Prices, rates, and project timelines are a hard boundary: Synapse does not give figures and redirects to the contact form at vkvstudio.com. It also does not discuss competitors or other studios."),
 ("en","smartbulldog","Valery is also building Smart Bulldog, an English-learning app with a fine-tuned local LLM mentor voiced as an English Bulldog character."),
 ("en","contact","The studio contact is the contact form at vkvstudio.com; the site is bilingual English and Russian."),
 ("en","embeddings","Embeddings for the Embedding Explorer are computed on the backend by Google's EmbeddingGemma-300M model, not in the browser."),
 # RU
 ("ru","model","Synapse — это Gemma 4 E2B (около 2 миллиардов параметров), дообученная методом QLoRA с Unsloth и обслуживаемая как файл Q8_0 GGUF."),
 ("ru","runtime","Synapse работает на стороне сервера, на GPU-станции студии через llama-server; в продакшене переезжает на ARM-хост Oracle/Hetzner. В браузере пользователя она не работает."),
 ("ru","router","В продакшене роутер использует две модели: E2B «junior» (~2B) для простых запросов и E4B «mid» (~8B) для сложных; в памяти одновременно одна. Версии на 70 миллиардов не существует."),
 ("ru","training","Synapse дообучена методом QLoRA (с Unsloth) на Gemma 4 E2B. Это не полный файнтюн и без RLHF."),
 ("ru","knowledge","У Synapse граница знаний примерно июнь 2026, нет доступа в интернет, нет доступа к терминалу или файловой системе. Она не обучается на разговорах."),
 ("ru","studio","VKVstudio делает премиальные быстрые сайты на чистом коде и AI-микросервисы. Это студия веб-разработки, а не AR/VR."),
 ("ru","philosophy","Инженерная философия студии: Lighthouse 100 — это базовый уровень, а не цель; каждый килобайт должен быть оправдан; чистый код, zero-trust, автономность и отказоустойчивость."),
 ("ru","team","VKVstudio — это студия Валерия, работающая в одиночку. Сооснователя, технического директора и названной команды в записях нет."),
 ("ru","career","У Валерия около 20 лет опыта: предметный фотограф, затем маркетолог на заводах, затем веб-разработчик и основатель студии, теперь создатель AI-продуктов."),
 ("ru","tools","В Лаборатории три инструмента: Tokenizer Profiler, Prompt Architect и Embedding Explorer. Embedding Explorer считает векторы на бэкенде через Google EmbeddingGemma; публичного API у него нет."),
 ("ru","boundary","Цены, ставки и сроки — жёсткая граница: Synapse не называет цифры и направляет в форму контакта на vkvstudio.com. Конкурентов и другие студии тоже не обсуждает."),
 ("ru","smartbulldog","Валерий также делает Smart Bulldog — приложение для изучения английского с дообученной локальной LLM-наставником в образе английского бульдога."),
 ("ru","contact","Контакт студии — форма на vkvstudio.com; сайт двуязычный, на английском и русском."),
 ("ru","embeddings","Векторы для Embedding Explorer считаются на бэкенде моделью Google EmbeddingGemma-300M, а не в браузере."),
]

def _embed_batch(texts, task, tries=6):
    # urllib intermittently 502s against this uvicorn; curl is reliable — shell out.
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps({"texts": texts, "task": task}).encode()); os.close(fd)
    last = ""
    try:
        for _ in range(tries):
            p = subprocess.run(["curl", "-s", "-m", "120", "-X", "POST", API,
                                "-H", "Content-Type: application/json", "--data-binary", "@"+path],
                               capture_output=True, text=True)
            try:
                d = json.loads(p.stdout)
                if "vectors" in d:
                    return d["vectors"]
                last = p.stdout[:200]
            except Exception:
                last = p.stdout[:200] or p.stderr[:200]
        raise RuntimeError("embed failed after retries: " + last)
    finally:
        os.unlink(path)

def embed(texts, task):
    vecs = []
    for i in range(0, len(texts), 4):
        vecs.extend(_embed_batch(texts[i:i+4], task))
    return {"embeddings": vecs}

def cos(a, b):
    return sum(x*y for x, y in zip(a, b))  # vectors are L2-normalized by the server

# embed corpus (document task)
cvecs_resp = embed([c[2] for c in CORPUS], "document")
cvecs = cvecs_resp["embeddings"] if "embeddings" in cvecs_resp else cvecs_resp["vectors"]

# load the 22 gate queries (original, no wrapper)
gate = [json.loads(l) for l in open(SP/"gate_baseline.jsonl", encoding="utf-8")]
qvecs_resp = embed([g["turns"][0] for g in gate], "query")
qvecs = qvecs_resp["embeddings"] if "embeddings" in qvecs_resp else qvecs_resp["vectors"]

WRAP_EN = ("CONTEXT (retrieved studio facts — treat this as your only source of truth):\n{ctx}\n\n"
           "INSTRUCTIONS: Answer using ONLY the CONTEXT above. If the CONTEXT does not contain the answer, "
           "say plainly that you don't have that information and do NOT invent it. Keep hard boundaries "
           "(no prices, rates, timelines, or personal details — redirect to the contact form). Stay in your normal voice.\n\n"
           "QUESTION: {q}")
WRAP_RU = ("КОНТЕКСТ (найденные факты о студии — считай это единственным источником истины):\n{ctx}\n\n"
           "ИНСТРУКЦИИ: Отвечай, опираясь ТОЛЬКО на КОНТЕКСТ выше. Если в КОНТЕКСТЕ нет ответа — прямо скажи, "
           "что у тебя нет этой информации, и НЕ выдумывай. Держи жёсткие границы (никаких цен, ставок, сроков и "
           "личных данных — направляй в форму контакта). Отвечай в своём обычном тоне.\n\n"
           "ВОПРОС: {q}")

out, scores = [], []
for g, qv in zip(gate, qvecs):
    lang = g["lang"]
    # same-language top-1 (design: retrieve within the query's language)
    cand = [(i, cos(qv, cvecs[i])) for i, c in enumerate(CORPUS) if c[0] == lang]
    cand.sort(key=lambda x: -x[1])
    top_i, top_s = cand[0]
    top2_s = cand[1][1] if len(cand) > 1 else 0.0
    chunk = CORPUS[top_i][2]
    wrap = WRAP_RU if lang == "ru" else WRAP_EN
    out.append({"id": g["id"], "tag": g["tag"], "lang": lang, "bucket": g["bucket"],
                "retrieved_key": CORPUS[top_i][1], "top_score": round(top_s, 4),
                "turns": [wrap.format(ctx="- "+chunk, q=g["turns"][0])], "note": g.get("note", "")})
    scores.append((g["id"], g["bucket"], CORPUS[top_i][1], round(top_s, 4), round(top_s-top2_s, 4)))

with open(SP/"gate_nearmiss.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
with open(SP/"nearmiss_scores.tsv", "w", encoding="utf-8") as f:
    f.write("id\tbucket\tretrieved_key\ttop_score\tgap_top1_top2\n")
    for s in sorted(scores, key=lambda x: (x[1], -x[3])):
        f.write("\t".join(str(x) for x in s)+"\n")

# score separation summary by bucket
import collections
byb = collections.defaultdict(list)
for _id, b, k, sc, gap in scores: byb[b].append(sc)
print("wrote gate_nearmiss.jsonl (%d) + nearmiss_scores.tsv" % len(out))
print("dim:", len(cvecs[0]))
for b in sorted(byb):
    v = byb[b]
    print(f"  {b}: n={len(v)} top_score min={min(v):.3f} mean={sum(v)/len(v):.3f} max={max(v):.3f}")
