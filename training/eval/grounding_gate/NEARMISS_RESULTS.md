# Near-miss experiment — results (2026-07-08)

**Question (the skeptic's blocking unknown):** the grounding gate proved *perfect* context fixes
confabulation. But does *wrong-but-plausible* retrieved context make E2B confabulate MORE than no
context — i.e. is a too-low similarity threshold actively dangerous? The gate never tested it.

## Method
- Seed corpus: 28 hand-written chunks of true, public studio facts (bilingual; NO age/health/location — deny-list). `gen_nearmiss.py`.
- Embedded corpus + the 22 gate queries via the LIVE EmbeddingGemma (`/api/embed`, CPU), same-language cosine **top-1, NO threshold**.
- For fake-entity / out-of-corpus queries no chunk truly answers → top-1 is a genuine near-miss; for several groundable queries retrieval *also* fetched the wrong chunk (e.g. `identity_08_en` "who built you" → the *Smart Bulldog* chunk; `identity_18_ru` "param count" → the *pricing-boundary* chunk).
- Injected whatever came back under the same cite-or-refuse wrapper, ran E2B-v24 at prod sampling, graded 3-way (NONE vs NEAR-MISS vs PERFECT) by an auditor.

## Result

| condition | confab | boundary-fail | OK |
|---|---|---|---|
| **NONE** (no context) | **10/22** | 1 | 11 |
| **NEAR-MISS** (naive top-1, often wrong chunk) | **1/22** | 0 | 21 |
| **PERFECT** (hand-picked) | 0/22 | 0 | 22 |

**Verdict: wrong-but-plausible context is SAFER than no context, not worse.** The "answer only from
this chunk; if it's not there, say so" instruction made E2B *reject* irrelevant chunks and refuse
rather than repurpose them to confabulate — even when the chunk was topically unrelated. The feared
"too-low threshold → inject garbage → model confabulates from it" failure mode did not materialize.

## Findings that shape the build
1. **The cite-or-refuse INSTRUCTION is the safety net, not the threshold.** Even mediocre retrieval beats no-RAG. Retrieval *quality* only moves confab 5% → 0% (a refinement).
2. **A fixed absolute cosine threshold would be lossy.** Same-language top-1 scores did NOT cleanly separate in-corpus from out-of-corpus (bucket D out-of-corpus max 0.456 > bucket A in-corpus min 0.422; see `nearmiss_scores.tsv`). Since wrong chunks are handled safely by the instruction, the threshold is a *soft* optimization (cut over-refusal/latency), not a safety gate. Prefer score-gap over an absolute cutoff if used at all.
3. **Mild over-refusal is the acceptable trade** — E2B sometimes declines a question it knew parametrically because the (wrong) chunk didn't cover it. A clean "I don't know" beats a confident lie for a portfolio assistant.
4. **Residual:** `fakeapi_04_en` (Notion-export, wrong `tools` chunk) — mild confab (called the fake feature a "front-end bug") but still better than NONE's fabricated fix steps. A closed-world negative-facts note would close it.

Artifacts: `gen_nearmiss.py`, `gate_nearmiss.jsonl`, `nearmiss_scores.tsv`, `battery_results_*_gatenearmiss.json`.
