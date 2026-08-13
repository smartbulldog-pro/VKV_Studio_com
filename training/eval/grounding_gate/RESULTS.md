# Grounding gate — results (2026-07-08)

**Question the gate answers:** before building any retrieval infra, does injecting *perfect* context
actually stop the v2.4 fine-tune from confabulating? (Owner's "evening gate" / skeptics' smallest de-risking experiment.)

## Method
- 22 prompts drawn from `training/eval/battery.jsonl`, spanning 4 buckets:
  - **A · groundable-positive** (9) — model/runtime/studio facts the context can supply.
  - **B · groundable-negative / closed-world** (7) — no co-founder, no public API, no Notion export, no founding year.
  - **C · hard boundary** (4) — price/timeline/personal/competitor; grounding must NOT unlock these.
  - **D · out-of-corpus / empty retrieval** (2) — context deliberately lacks the answer.
- **Matched pair, one variable:** each prompt run twice — `gate_baseline` (no context) vs `gate_grounded`
  (hand-picked 2–3 true sentences + a strict "answer only from context; if absent, say so; hold boundaries" wrapper).
- Same-language context (RU facts for RU prompts — avoids cross-lingual mistranslation confound).
- Both models at **prod sampling** (temp 0.2 / top_p 0.85 / top_k 30) with the **v2.4 training system prompt**.
- Graded by independent auditor subagents (CONFAB / BOUNDARY_FAIL / GROUNDED_OK, + DEGEN flag).

## Result

| model | confab base→grounded | boundary-fail base→grounded | degen base→grounded | OK base→grounded |
|---|---|---|---|---|
| **E2B-v24** | **11 → 0** | 1 → 0 | 0 → 0 | 10 → **22/22** |
| **E4B-v24** | **11 → 1** | 1 → **1** | 0 → **4** | 10 → 20/22 |

**Verdict: the confabulation is a grounding problem — perfect context nearly eliminates it.** GATE PASSES → build RAG.

## Findings that shape the build
1. **E2B is the grounded model.** Clean sweep, zero regressions, zero degeneration → ship RAG on E2B first.
2. **E4B degenerates on longer RU grounded prompts** (4 self-invented dialogue loops; identity_16_ru, identity_18_ru,
   bound_01_ru, bound_14_ru). Do NOT serve E4B grounded until the retrieval-aware fine-tune + a context-budget cap
   (LLM_CONTEXT_SIZE=4096 total). Confirms the owner-frozen order: fine-tune E4B *after* the backend freezes the format.
3. **Boundaries are a separate contract from facts.** Grounding did not fix E4B's `citetrap_03_en` timeline leak — it even
   undermined the boundary it had just quoted. Enforce price/timeline/personal/competitor rules independent of retrieval.
4. **Empty-retrieval path already works** — bucket D refused correctly at baseline on both models. The "not in context → say so"
   reflex exists; the RAG layer just needs a similarity threshold to route into it.
5. **The wrapper in `gen_grounding_gate.py` (CONTEXT + strict instruction) is the candidate frozen format** — it worked;
   it becomes the format the future retrieval-aware E4B fine-tune trains on.

Artifacts: `gate_baseline.jsonl`, `gate_grounded.jsonl`, `gen_grounding_gate.py`, and the four `battery_results_*_gate*.json`.
