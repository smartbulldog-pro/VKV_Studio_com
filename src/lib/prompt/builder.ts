import {
  tokenize,
  type ModelId,
  MODELS as TOKENIZER_MODELS,
  OPENAI_CHATML_TOKENS_PER_MESSAGE,
  OPENAI_CHATML_REPLY_PRIMING_TOKENS,
  API_CHAT_OVERHEAD_ESTIMATE_TOKENS,
  type ChatOverheadExactness,
  type Accuracy,
  type FallbackReason,
} from '../tokenizer/engine';

export type BlockRole = 'system' | 'user' | 'assistant' | 'context' | 'tools';

/** Hard cap on a single block's content length. Without one, pasting a huge
 *  (50MB+) file into a block freezes the tokenizer/highlight-overlay/DnD UI —
 *  100k chars comfortably covers any real prompt (novels are ~500k chars,
 *  the *content* of a prompt block is rarely more than a few thousand) while
 *  keeping the editor responsive. */
export const MAX_BLOCK_CONTENT_LENGTH = 100_000;

export interface PromptBlock {
  id: string;
  role: BlockRole;
  content: string;
  tokens: number;
  collapsed: boolean;
  /** Whether this block is treated as a stable, cache-eligible prefix segment
   *  (e.g. a frozen system prompt or RAG scaffolding) vs. volatile per-turn
   *  content. Defaults to `true` for `system`/`context` roles, `false` for
   *  `user`/`assistant` — see `defaultCacheableForRole`. */
  cacheable: boolean;
}

/** Sensible default for whether a newly-created block should be treated as
 *  cacheable: stable roles (system prompt, RAG/context scaffolding, tool
 *  definitions — all typically frozen between calls) default to cacheable;
 *  per-turn roles (user, assistant) default to not cacheable. */
export function defaultCacheableForRole(role: BlockRole): boolean {
  return role === 'system' || role === 'context' || role === 'tools';
}

/** Which provider's API surface a model's export shape / batch-pricing /
 *  cache-floor rules follow. Anthropic and OpenAI get dedicated export
 *  formats (see `exportAsAnthropicMessages` / `exportAsOpenAIChatCompletions`
 *  / `exportAsOpenAIResponsesAPI`); every other provider's models still
 *  export via the OpenAI Chat Completions shape, since Google/DeepSeek/
 *  Alibaba/Mistral/Zhipu all expose OpenAI-compatible endpoints in practice —
 *  that shape doubles as the generic/default rather than needing a distinct
 *  fourth "generic" format. */
export type Provider =
  | 'anthropic'
  | 'openai'
  | 'google'
  | 'deepseek'
  | 'alibaba'
  | 'mistral'
  | 'zhipu'
  | 'local';

export interface ModelConfig {
  id: string;
  name: string;
  contextWindow: number;
  inputPrice: number;
  outputPrice: number;
  /** Whether the provider supports prompt caching for this model at all. */
  supportsCaching: boolean;
  /** Multiplier applied to `inputPrice` for a cache *write* (first call that
   *  populates the cache for a given prefix). */
  cacheWriteMultiplier: number;
  /** Multiplier applied to `inputPrice` for a cache *read* (subsequent calls
   *  that hit an already-populated cache). */
  cacheReadMultiplier: number;
  /** Provider whose API conventions this model's exports/pricing follow. */
  provider: Provider;
  /**
   * Minimum token count a cacheable block must reach before the provider will
   * actually cache it — Anthropic documents ≈1024 tokens for Sonnet-tier
   * models and ≈4096 for Opus/Haiku-tier models; OpenAI documents 1024 for
   * automatic prompt caching. Google/DeepSeek/Alibaba/Mistral/Zhipu have no
   * minimum documented here as of 2026-07-09 in the sources already cited in
   * this file's price comments — a conservative 1024 is used for them rather
   * than fabricating a provider-specific figure; treat those rows' floor as a
   * placeholder, not a verified number.
   */
  minCacheTokens: number;
  /**
   * Multiplier applied to the FINAL (already cache-discounted) call cost when
   * the user opts into the provider's batch/asynchronous API — 0.5 for
   * Anthropic's Message Batches API and OpenAI's Batch API (both documented
   * −50% on input+output). Omitted (`undefined`) for providers with no
   * documented batch discount here, rather than guessing one.
   */
  batchDiscount?: number;
  /**
   * True when this model has hidden/billable "reasoning" tokens that can
   * make real output cost several times the estimate shown here. Derived
   * from `tokenizer/engine.ts`'s `ModelInfo.hiddenReasoning` — the single
   * source of truth — via the `.map()` below, so the two tools can never
   * silently disagree about which models this applies to.
   */
  reasoningCapable: boolean;
}

// Cache multipliers: live-verified 2026-07-09 against openrouter.ai/api/v1/models
// raw pricing objects (input_cache_read / input_cache_write ÷ prompt price) where
// noted below; rows without a live-verify note keep the prior documented estimate.
// Prices reconciled with .system/models_june_2026.md (re-dated 2026-07-09).
//
// `minCacheTokens` / `batchDiscount` (added Phase B, 2026-07-09): Anthropic
// documents a ≈1024-token minimum cacheable prefix for Sonnet-tier models and
// ≈4096 tokens for Opus/Haiku/Fable-tier models; OpenAI documents 1024 for its
// automatic prompt caching. Both document a −50% batch-API discount. Google/
// DeepSeek/Alibaba/Mistral/Zhipu have no minimum documented in the sources
// already cited per-row below, so they get a conservative 1024 placeholder
// (NOT a verified number) and no batch discount (undefined — not guessed).
//
// METHODOLOGY CORRECTION (2026-08-09): the 2026-07-09 pass above sourced
// prices from OpenRouter's top-level `pricing` field, which is whichever
// endpoint OpenRouter currently designates as "default" for a model id —
// frequently a RESELLER's rate, not the provider's own list price. That
// silently mispriced several rows below (glm-5.2, deepseek-v4-flash; and
// mistral-large-3 turned out to be a DIFFERENT model's price — Large 2's, not
// a stale-but-correct number for Large 3). Every row was re-verified today
// directly against each provider's own pricing page; row comments below say
// "REPRICED 2026-08-09" or "corrected" wherever a number changed. Going
// forward, prefer `GET /api/v1/models/{id}/endpoints` (a specific endpoint's
// own pricing) over the top-level `pricing` field on `GET /api/v1/models` if
// OpenRouter is used again — or just read the provider's own page.
type ModelConfigBase = Omit<ModelConfig, 'reasoningCapable'>;
const MODELS_BASE: ModelConfigBase[] = [
  // Google
  // Renamed 2026-08-09: id/name were 'gemini-3.5-pro' / "Gemini 3.1 Pro
  // Preview". "Gemini 3.5 Pro" does not exist — absent from both Google's
  // pricing page and its models page, and Google's own "Pro" tier page shows
  // a "coming soon" badge, not a shipped model. The $2.00/$12.00 prices and
  // 1,048,576 context here were always Gemini 3.1 Pro's numbers (see the
  // superseded comment this replaces), so the id/name are corrected to match
  // the model these numbers actually belong to; "Preview" is dropped since
  // 3.1 Pro is GA. cacheReadMultiplier corrected 0.25 → 0.10 in the same pass.
  // RESOLVED 2026-08-13. This note used to warn that `tokenizer/engine.ts` and
  // `inference/main.py` still keyed the model as 'gemini-3.5-pro', so the id
  // here would not join against `TOKENIZER_MODELS`. Both have since caught up —
  // engine.ts keys it 'gemini-3.1-pro' with an `api` backend, and
  // `map_model_name` passes the id through untouched — and the join is now
  // guarded by tests/unit/model-roster.test.ts. Left as a record of why the
  // rename happened, not as an open warning: read as current, it sent you
  // looking for a mismatch that no longer exists.
  { id: 'gemini-3.1-pro',   name: 'Gemini 3.1 Pro',       contextWindow: 1048576, inputPrice: 2.00,  outputPrice: 12.00, supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.10, provider: 'google', minCacheTokens: 1024 },
  // Repriced 2026-07-09 (was $0.15/$0.60 — ~10x under live). Live-verified via
  // openrouter.ai (google/gemini-3.5-flash): prompt $0.0000015, completion
  // $0.000009 → $1.50/$9.00 per 1M. cacheReadMultiplier from the same raw
  // pricing object: input_cache_read $0.00000015 ÷ prompt = 0.10. The raw
  // object's "input_cache_write" ($0.0000000833/tok) is far below the prompt
  // price and reads as a per-hour storage-style fee, not a first-call write
  // premium in this field's sense — kept at the documented 1.0 (full price to
  // populate the cache) rather than reinterpreting an ambiguous live number.
  { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash',     contextWindow: 1048576, inputPrice: 1.50,  outputPrice: 9.00,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.10, provider: 'google', minCacheTokens: 1024 },
  // New 2026-08-09. Source: ai.google.dev/gemini-api/docs/pricing. No batch
  // discount documented for this row (same as the other Google/DeepSeek/
  // Alibaba/Mistral/Zhipu rows here) — omitted rather than guessed.
  { id: 'gemini-3.6-flash', name: 'Gemini 3.6 Flash',     contextWindow: 1048576, inputPrice: 1.50,  outputPrice: 7.50,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.10, provider: 'google', minCacheTokens: 1024 },
  { id: 'gemma-4-e2b',      name: 'Gemma 4 E2B (local)',  contextWindow: 32000,   inputPrice: 0,     outputPrice: 0,     supportsCaching: false, cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 1.0, provider: 'local', minCacheTokens: 1024 },
  // Anthropic
  // Fable 5 priced above Opus intentionally — it is Anthropic's flagship model,
  // above the Opus tier, per .system/models_june_2026.md ($10/$50) and current
  // published pricing. Not a bug: Fable 5 > Opus 4.8 > Sonnet 5 is the correct order.
  { id: 'claude-fable-5',   name: 'Claude Fable 5',       contextWindow: 1000000, inputPrice: 10.00, outputPrice: 50.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'anthropic', minCacheTokens: 4096, batchDiscount: 0.5 },
  { id: 'claude-opus-4.8',  name: 'Claude Opus 4.8',      contextWindow: 1000000, inputPrice: 5.00,  outputPrice: 25.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'anthropic', minCacheTokens: 4096, batchDiscount: 0.5 },
  // New 2026-08-09. Source: platform.claude.com/docs/en/docs/about-claude/pricing.
  // Same $5.00/$25.00 @ 1M as claude-opus-4.8 above — kept as a separate row
  // rather than merged into/replacing it, since neither that source nor this
  // file's existing comments document claude-opus-4.8 as retired; re-check at
  // the next pass whether 4.8 is still a live model id.
  { id: 'claude-opus-5',    name: 'Claude Opus 5',        contextWindow: 1000000, inputPrice: 5.00,  outputPrice: 25.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'anthropic', minCacheTokens: 4096, batchDiscount: 0.5 },
  // Renamed from 'claude-sonnet-4.6' 2026-07-09. $2.00/$10.00 per 1M is INTRO
  // pricing that Anthropic's own announcement documents as ending 2026-08-31,
  // after which it steps to $3.00/$15.00 — re-verified 2026-08-09 (still ~3
  // weeks out from the cutover, no change to apply yet).
  // >>> REVISIT THIS ROW AFTER 2026-08-31 — price steps to $3.00/$15.00. <<<
  // Source: anthropic.com/news/claude-sonnet-5. Cache multipliers (write ÷
  // prompt = 1.25, read ÷ prompt = 0.10) live-verified 2026-07-09, unchanged.
  { id: 'claude-sonnet-5',  name: 'Claude Sonnet 5',      contextWindow: 1000000, inputPrice: 2.00,  outputPrice: 10.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'anthropic', minCacheTokens: 1024, batchDiscount: 0.5 },
  // Renamed id 2026-08-09: 'claude-haiku' is not a valid Anthropic model id
  // (a real API call would 400) — the current generation is Haiku 4.5, and
  // there is no "Haiku 5". Display name ("Claude Haiku") and the $1.00/$5.00
  // @ 200K pricing are unchanged. Source:
  // platform.claude.com/docs/en/docs/about-claude/pricing.
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5',     contextWindow: 200000,  inputPrice: 1.00,  outputPrice: 5.00,  supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'anthropic', minCacheTokens: 4096, batchDiscount: 0.5 },
  // OpenAI
  // CAVEAT (documented 2026-08-09 — schema below has no field for this):
  // GPT-5.5 and every GPT-5.6 model (sol/terra/luna) bill the ENTIRE request
  // at 2x the input price and 1.5x the output price once the prompt exceeds
  // 272,000 tokens — a long-context pricing-TIER switch, not a marginal
  // per-token surcharge past the threshold. `ModelConfig` has no field to
  // express a context-length-dependent price tier, and none is added here
  // (out of scope for this pass — a partial/wrong implementation would be
  // worse than an honest gap). Practical effect: `calculateCost` /
  // `calculateCachedCost` silently UNDER-estimate cost by roughly 2x input /
  // 1.5x output for any prompt over 272K tokens against every OpenAI row
  // below. Source: developers.openai.com/api/docs/pricing.
  { id: 'gpt-5.5',          name: 'GPT-5.5',              contextWindow: 1050000, inputPrice: 5.00,  outputPrice: 30.00, supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.10, provider: 'openai', minCacheTokens: 1024, batchDiscount: 0.5 },
  // cacheReadMultiplier corrected 0.5 → 0.10 2026-08-09 (re-verified against
  // developers.openai.com/api/docs/pricing): cached input is billed at
  // $0.50/M against the $5.00/M list price — a 0.10 multiplier of
  // `inputPrice` — not the pre-GPT-5 "50% off cached input" policy the old
  // 0.5 value assumed. Prices/context otherwise unchanged.
  { id: 'gpt-5.5-pro',      name: 'GPT-5.5 Pro',          contextWindow: 1050000, inputPrice: 30.00, outputPrice: 180.00, supportsCaching: true, cacheWriteMultiplier: 1.0, cacheReadMultiplier: 0.10, provider: 'openai', minCacheTokens: 1024, batchDiscount: 0.5 },
  // cacheReadMultiplier corrected 0.5 → 0.10 2026-08-09, same reasoning as the
  // gpt-5.5 row above (no cache fields were live-verified for this specific
  // row in the 2026-07-09 pass; now aligned to OpenAI's documented
  // platform-wide cached-input policy). Prices/context otherwise unchanged.
  //
  // GPT-5.6 (sol/terra/luna) added 2026-08-09 — absent from the catalog
  // checked 2026-07-09, now published at developers.openai.com/api/docs/pricing.
  // (A GPT-5.5 "mini" tier was also checked for 2026-07-09 and did not exist;
  // not re-checked today, not re-added here.) There is no GPT-5.6 "Pro" model
  // id: "pro" is a `reasoning.mode` flag on the SAME model id, not a distinct
  // model/price — do NOT add a `gpt-5.6-*-pro` row. All three share OpenAI's
  // documented cache/batch terms (cacheRead 0.10, cacheWrite 1.25,
  // batchDiscount 0.5) and the same >272K-token pricing-tier caveat above.
  { id: 'gpt-5.6-sol',      name: 'GPT-5.6 Sol',          contextWindow: 1050000, inputPrice: 5.00,  outputPrice: 30.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'openai', minCacheTokens: 1024, batchDiscount: 0.5 },
  { id: 'gpt-5.6-terra',    name: 'GPT-5.6 Terra',        contextWindow: 1050000, inputPrice: 2.00,  outputPrice: 12.00, supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'openai', minCacheTokens: 1024, batchDiscount: 0.5 },
  { id: 'gpt-5.6-luna',     name: 'GPT-5.6 Luna',         contextWindow: 1050000, inputPrice: 0.20,  outputPrice: 1.20,  supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'openai', minCacheTokens: 1024, batchDiscount: 0.5 },
  //
  // DeepSeek
  // Split into Pro/Flash 2026-07-09 (was one 'deepseek-v4' row). Both
  // live-verified via openrouter.ai raw pricing objects.
  // Pro: prompt $0.000000435, completion $0.00000087 → $0.44/$0.87 (rounded).
  // cacheReadMultiplier = input_cache_read ÷ prompt = 0.000000003625 ÷
  // 0.000000435 ≈ 0.0083 (DeepSeek's cache-hit discount is steeper than the
  // previously-documented 0.10 estimate — kept at full live precision here).
  { id: 'deepseek-v4-pro',  name: 'DeepSeek V4 Pro',      contextWindow: 1048576, inputPrice: 0.44,  outputPrice: 0.87,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.0083, provider: 'deepseek', minCacheTokens: 1024 },
  // Flash REPRICED 2026-08-09 (was $0.09/$0.18 — sourced 2026-07-09 from
  // OpenRouter's top-level `pricing` field, a reseller rate, not DeepSeek's
  // own price). Re-verified against api-docs.deepseek.com/quick_start/pricing:
  // $0.14/$0.28 per 1M. cacheReadMultiplier corrected 0.20 → 0.02 from the
  // same source (documented cache-hit price is $0.028/M against the $0.14/M
  // input price — the old 0.20 was computed off the wrong base price).
  { id: 'deepseek-v4-flash',name: 'DeepSeek V4 Flash',    contextWindow: 1048576, inputPrice: 0.14,  outputPrice: 0.28,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.02, provider: 'deepseek', minCacheTokens: 1024 },
  // Qwen — prices $1.25/$3.75 KEPT 2026-08-09, but flagged: re-verified
  // against alibabacloud.com/help/en/model-studio/qwen3-7-max, and $1.25/
  // $3.75 is a LIMITED-TIME 50%-off PROMOTIONAL rate on a $2.50/$7.50
  // Singapore-region list price — it will roughly double whenever the promo
  // ends, and Alibaba's page does not give an end date. Re-check this row
  // more often than the normal cadence. cacheReadMultiplier corrected 0.25 →
  // 0.10, cacheWriteMultiplier corrected 1.0 → 1.25, both per the same source.
  { id: 'qwen-3.7-max',     name: 'Qwen 3.7-Max',        contextWindow: 1000000, inputPrice: 1.25,  outputPrice: 3.75,  supportsCaching: true,  cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.10, provider: 'alibaba', minCacheTokens: 1024 },
  // Mistral Large 3 — REPRICED 2026-08-09. The previous $2.00/$6.00 (kept
  // through the 2026-07-09 pass as "corroborated but not primary-verified")
  // turned out to be Mistral LARGE 2's price — a different model, not a
  // stale-but-correct price for Large 3. Re-verified against
  // mistral.ai/pricing/api: Large 3 is $0.50/$1.50 per 1M. cacheReadMultiplier
  // corrected 0.25 → 0.10 from the same source. Context (262,144, widened
  // 2026-07-09 per Mistral's model card) unchanged.
  { id: 'mistral-large-3',  name: 'Mistral Large 3',      contextWindow: 262144,  inputPrice: 0.50,  outputPrice: 1.50,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.10, provider: 'mistral', minCacheTokens: 1024 },
  // Llama 4 Scout and GLM-5.2 tokenizers are wired in engine.ts (Tokenizer
  // Profiler). Llama 4 Scout is NOT priced here: it does not appear anywhere
  // in the live OpenRouter catalog as of 2026-07-09 (checked every meta-llama/
  // and llama-4/scout id), so it's omitted from cost estimation rather than
  // guessing a price. GLM-5.2 IS priced below since it was found live.
  //
  // Zhipu — REPRICED 2026-08-09. The previous $0.84/$2.64 (2026-07-09 pass,
  // sourced from OpenRouter's top-level `pricing` field) was a reseller
  // quote — Zhipu's own price never actually changed. Re-verified against
  // docs.z.ai/guides/overview/pricing: $1.40/$4.40 per 1M, 1,048,576 ctx
  // (unchanged). cacheReadMultiplier (0.19, from the same now-known-unreliable
  // OpenRouter listing) was NOT re-verified against the primary source in
  // this pass — treat it as unconfirmed, not corrected.
  { id: 'glm-5.2',          name: 'GLM-5.2',              contextWindow: 1048576, inputPrice: 1.40,  outputPrice: 4.40,  supportsCaching: true,  cacheWriteMultiplier: 1.0,  cacheReadMultiplier: 0.19, provider: 'zhipu', minCacheTokens: 1024 },
  // Moonshot Kimi K2.7-Code was checked (live-priced on OpenRouter: $0.74/$3.50)
  // but has NO anonymously-fetchable HF fast-tokenizer (tokenizer.json 404 across
  // every Kimi K2 family repo checked — it ships a custom tiktoken.model +
  // custom_code tokenizer transformers.js cannot load), so it's left out of both
  // the tokenizer tool and this pricing table per the same-tokenizer-required
  // rule. Listed doc-only in .system/models_june_2026.md.
];

/**
 * Public model roster. `reasoningCapable` is derived here (not stored on
 * `MODELS_BASE`) from `tokenizer/engine.ts`'s `ModelInfo.hiddenReasoning` —
 * that flag is the single source of truth for which models have hidden
 * billable reasoning tokens, so this `.map()` is the only place the two
 * tools' rosters could silently drift apart, and it can't (same boolean,
 * read live, every time `MODELS` is imported).
 */
export const MODELS: ModelConfig[] = MODELS_BASE.map((m) => ({
  ...m,
  reasoningCapable: TOKENIZER_MODELS[m.id as ModelId]?.hiddenReasoning === true,
}));

// ─── Template Presets ─────────────────────────────────────────────────────────

export interface TemplatePreset {
  id: string;
  name: string;
  description: string;
  blocks: Omit<PromptBlock, 'id' | 'tokens'>[];
}

export const TEMPLATES: TemplatePreset[] = [
  {
    id: 'chat-assistant',
    name: 'Chat Assistant',
    description: 'General-purpose conversational assistant',
    blocks: [
      {
        role: 'system',
        content: 'You are a helpful, concise, and accurate AI assistant. Answer questions clearly and professionally. If you are unsure, say so.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Hello! Can you help me with {{task}}?',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  {
    id: 'code-review',
    name: 'Code Review',
    description: 'Code quality and security review',
    blocks: [
      {
        role: 'system',
        content: 'You are a Senior Software Engineer conducting a code review. Focus on: correctness, performance, security, and readability. Be specific and actionable.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'context',
        content: '{{code_snippet}}',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Please review this {{language}} code. Pay special attention to {{focus_area}}.',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  {
    id: 'translation',
    name: 'Translation',
    description: 'Accurate multilingual translation',
    blocks: [
      {
        role: 'system',
        content: 'You are a professional translator. Translate the provided text accurately, preserving tone, style, and nuance. Do not add explanations unless explicitly asked.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Translate the following from {{source_lang}} to {{target_lang}}:\n\n{{text}}',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  {
    id: 'rag-pipeline',
    name: 'RAG Pipeline',
    description: 'Retrieval-Augmented Generation with context grounding',
    blocks: [
      {
        role: 'system',
        content: 'You are a precise AI assistant that answers questions exclusively based on the provided context. If the context does not contain sufficient information, state that clearly. Do not hallucinate facts.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'context',
        content: '{{retrieved_chunks}}',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: '{{user_question}}',
        collapsed: false,
        cacheable: false,
      },
      {
        role: 'assistant',
        content: 'Based on the provided context,',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  // Few-shot: not a new mechanism — a preset built entirely from the existing
  // user/assistant block types, just several example turns in a row before
  // the real question.
  {
    id: 'few-shot',
    name: 'Few-Shot Examples',
    description: 'System prompt + worked user/assistant example pairs before the real question',
    blocks: [
      {
        role: 'system',
        content: 'You are a classifier. Follow the exact pattern shown in the examples below, then classify the final input the same way.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Input: "This product broke after one day, total waste of money."',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'assistant',
        content: 'Sentiment: negative',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Input: "Works exactly as described, very happy with it."',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'assistant',
        content: 'Sentiment: positive',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: 'Input: "{{input_text}}"',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  // Tool-definitions example — demonstrates the 'tools' block role. The JSON
  // below uses Anthropic's tool-schema field names (`input_schema`) as a
  // widely-recognized example shape; export functions intentionally do NOT
  // translate between providers' tool-schema dialects (e.g. OpenAI's
  // `{type:'function', function:{...,parameters}}` wrapper) — they pass
  // whatever JSON is in the block through as-is, so swap this example for
  // your target provider's real shape before using it against that API.
  {
    id: 'tool-use',
    name: 'Tool Use',
    description: 'System prompt + a tool-definitions block + a question that would invoke it',
    blocks: [
      {
        role: 'system',
        content: 'You are an assistant with access to tools. Use the get_weather tool when the user asks about current weather; otherwise answer directly.',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'tools',
        content: JSON.stringify(
          [
            {
              name: 'get_weather',
              description: 'Get the current weather for a given location',
              input_schema: {
                type: 'object',
                properties: {
                  location: { type: 'string', description: 'City and country, e.g. "Paris, France"' },
                  unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
                },
                required: ['location'],
              },
            },
          ],
          null,
          2
        ),
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: "What's the weather like in {{city}} right now?",
        collapsed: false,
        cacheable: false,
      },
    ],
  },
  {
    id: 'custom',
    name: 'Custom',
    description: 'Start from a blank slate',
    blocks: [
      {
        role: 'system',
        content: '',
        collapsed: false,
        cacheable: true,
      },
      {
        role: 'user',
        content: '',
        collapsed: false,
        cacheable: false,
      },
    ],
  },
];

// ─── Variable Extraction & Substitution ───────────────────────────────────────

/** Extract all unique {{variable}} names from a set of blocks */
export function extractVariables(blocks: PromptBlock[]): string[] {
  const pattern = /\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}/g;
  const seen = new Set<string>();
  for (const block of blocks) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(block.content)) !== null) {
      seen.add(match[1]);
    }
  }
  return Array.from(seen);
}

/** Replace {{variable}} occurrences with provided values */
export function applyVariables(text: string, values: Record<string, string>): string {
  return text.replace(/\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}/g, (_, name) => {
    return values[name] !== undefined ? values[name] : `{{${name}}}`;
  });
}

/** Apply variable substitution to all blocks, returning new content strings */
export function applyVariablesToBlocks(blocks: PromptBlock[], values: Record<string, string>): PromptBlock[] {
  return blocks.map(b => ({ ...b, content: applyVariables(b.content, values) }));
}

// ─── Core Functions ───────────────────────────────────────────────────────────

/**
 * Tokenizes `text` for `model`, optionally cancellable via `signal` — threads
 * straight through to `tokenize()`'s own `signal` option (only meaningful for
 * API-backed models; a no-op for the synchronous on-device tiktoken/
 * transformers backends). Callers doing per-keystroke recomputation (e.g.
 * PromptApp's debounced block tokenizer) should abort the previous call's
 * signal before starting a new one, mirroring the Tokenizer tool's
 * `activeController` pattern — otherwise a slow, now-stale API round-trip can
 * resolve after a newer one and briefly show wrong counts.
 */
export interface BlockCount {
  tokens: number;
  /**
   * How the number was produced. This used to be discarded — `countTokens`
   * returned a bare number — so the Prompt Architect showed a token count, and
   * a dollar cost derived from it, without ever saying whether the count came
   * from the provider's own tokenizer or from the cl100k stand-in. On a tool
   * whose entire pitch is honest numbers that was the wrong thing to throw
   * away, so it is threaded through now.
   */
  accuracy: Accuracy;
  /** Present only when `accuracy` is 'approx'. */
  fallbackReason?: FallbackReason;
}

export async function countTokens(
  text: string,
  model: ModelConfig,
  opts?: { signal?: AbortSignal; allowNetwork?: boolean }
): Promise<BlockCount> {
  // An empty block is exactly zero tokens in every tokenizer — no approximation
  // involved, so this does not need to be labelled as one.
  if (!text.trim()) return { tokens: 0, accuracy: 'native' };
  const result = await tokenize(text, model.id as ModelId, opts);
  return {
    tokens: result.totalTokens,
    accuracy: result.accuracy,
    fallbackReason: result.fallbackReason,
  };
}

/** Result of `chatWrappingOverhead` — the estimated overhead PLUS an honest
 *  tier for how it was produced, so the UI never claims more precision than
 *  it has (mirrors `tokenizer/engine.ts`'s `ChatOverhead`/`ChatOverheadExactness`). */
export interface ChatWrappingOverheadResult {
  overhead: number;
  exactness: ChatOverheadExactness;
}

// Chat wrapping overhead. A prompt sent to a chat API is not just the raw
// block text: each message carries role/wrapper tokens and the reply is
// primed once. Previously this applied OpenAI's documented ~3 tokens/message
// + 3 reply-priming-token ChatML formula to EVERY provider silently — honest
// only for OpenAI's own tiktoken-backed models. Fixed here: the formula is
// only applied (tier 'documented') when the selected model's tokenizer
// backend actually IS tiktoken (i.e. it's an OpenAI model, per
// `tokenizer/engine.ts`'s `MODELS[id].backend.type`); every other backend
// (Anthropic/Google API, or a self-hosted transformers model whose real
// per-conversation chat-template overhead isn't cheaply computable from here)
// falls back to a conservative flat-per-message placeholder tagged
// 'estimated' — same order of magnitude as the Tokenizer tool's single-turn
// `API_CHAT_OVERHEAD_ESTIMATE_TOKENS`, reused directly rather than
// reinvented, but honestly never upgraded to 'documented' or 'native'.
export function chatWrappingOverhead(messageCount: number, model: ModelConfig): ChatWrappingOverheadResult {
  if (messageCount <= 0) return { overhead: 0, exactness: 'documented' };

  const backend = TOKENIZER_MODELS[model.id as ModelId]?.backend;
  if (backend?.type === 'tiktoken') {
    return {
      overhead: messageCount * OPENAI_CHATML_TOKENS_PER_MESSAGE + OPENAI_CHATML_REPLY_PRIMING_TOKENS,
      exactness: 'documented',
    };
  }

  return {
    overhead: messageCount * API_CHAT_OVERHEAD_ESTIMATE_TOKENS,
    exactness: 'estimated',
  };
}

export function calculateCost(tokens: number, model: ModelConfig): number {
  return (tokens / 1_000_000) * model.inputPrice;
}

export function calculateOutputCost(tokens: number, model: ModelConfig): number {
  return (tokens / 1_000_000) * model.outputPrice;
}

export interface CachedCostResult {
  /** Total cost of the first call, where cacheable tokens are written to cache. */
  firstCall: number;
  /** Total cost of a later call, where cacheable tokens are read from cache. */
  cachedCall: number;
  /** Percentage saved on a cached call vs. the first call (0 when the model
   *  doesn't support caching, or when firstCall is 0). */
  savingsPct: number;
}

/**
 * Cache-aware cost model: cacheable tokens (e.g. a stable system prompt or
 * RAG scaffolding) are billed at a cache-write rate on the first call and a
 * much cheaper cache-read rate on every subsequent call; non-cacheable
 * (fresh, per-turn) tokens are always billed at the plain input rate; output
 * tokens are billed the same either way. `cacheableTokens` here must already
 * be floor-filtered (see `evaluateCacheEligibility`) — this function has no
 * opinion on the provider's minimum cacheable prefix, only on how to price
 * whatever token count it's given.
 *
 * `batchMultiplier` (default 1 = no discount) models opting into the
 * provider's batch/async API: Anthropic's Message Batches API and OpenAI's
 * Batch API both apply their −50% discount on top of an already
 * cache-discounted price (per Anthropic's docs, batch + cache stack
 * multiplicatively, not additively) — applied here as a final multiplier
 * over the whole call cost, cache savings included.
 */
export function calculateCachedCost(
  cacheableTokens: number,
  freshTokens: number,
  outputTokens: number,
  model: ModelConfig,
  batchMultiplier = 1
): CachedCostResult {
  const outputCost = calculateOutputCost(outputTokens, model);
  const freshCost = (freshTokens / 1_000_000) * model.inputPrice;

  if (!model.supportsCaching) {
    const flatCost = ((cacheableTokens / 1_000_000) * model.inputPrice + freshCost + outputCost) * batchMultiplier;
    return { firstCall: flatCost, cachedCall: flatCost, savingsPct: 0 };
  }

  const writeCost = (cacheableTokens / 1_000_000) * model.inputPrice * model.cacheWriteMultiplier;
  const readCost = (cacheableTokens / 1_000_000) * model.inputPrice * model.cacheReadMultiplier;

  const firstCall = (writeCost + freshCost + outputCost) * batchMultiplier;
  const cachedCall = (readCost + freshCost + outputCost) * batchMultiplier;
  const savingsPct = firstCall > 0 ? ((firstCall - cachedCall) / firstCall) * 100 : 0;

  return { firstCall, cachedCall, savingsPct };
}

// ─── Cache floor (item 2) ──────────────────────────────────────────────────

export interface CacheEligibility {
  /** Sum of tokens across blocks marked cacheable AND at/above the model's
   *  `minCacheTokens` floor — these are the tokens that actually get cached. */
  eligibleCacheableTokens: number;
  /** Sum of tokens across blocks marked cacheable but BELOW the floor — the
   *  provider would silently refuse to cache these, so they're billed as
   *  fresh input on every call instead of contributing any cache savings. */
  belowFloorTokens: number;
  /** IDs of the cacheable-but-below-floor blocks, for graying their cache
   *  badge in the UI. */
  belowFloorBlockIds: string[];
  /** Count of blocks that actually clear the floor (used for the Anthropic
   *  breakpoint-limit warning below). */
  eligibleBlockCount: number;
  /** True when `eligibleBlockCount` exceeds Anthropic's documented limit of
   *  4 cache breakpoints per request. Only ever true for `provider ===
   *  'anthropic'` — other providers document different (or no) breakpoint
   *  limits, so this is left false for them rather than guessing a number. */
  exceedsBreakpointLimit: boolean;
}

const ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4;

/**
 * IDs of blocks that are cache-eligible (cacheable AND at/above `model`'s
 * `minCacheTokens` floor) AND — for Anthropic only — rank within the
 * Messages API's documented 4-cache-breakpoint-per-request limit, counted in
 * `blocks` prompt order. Non-Anthropic providers have no documented
 * breakpoint ceiling here, so every floor-clearing block qualifies for them.
 *
 * This is the single source of truth for "which blocks are REALLY cached",
 * shared by `evaluateCacheEligibility` (so the savings estimate is capped
 * the same way) and `exportAsAnthropicMessages` (so the same blocks are the
 * ones that actually receive a `cache_control` marker) — the P0 fix this
 * supports was that the caching panel showed savings the export never
 * actually encoded; capping both off the same function is what keeps them
 * from silently drifting apart again.
 */
function cacheEligibleBlockIds(blocks: PromptBlock[], model: ModelConfig): Set<string> {
  const ids = new Set<string>();
  let eligibleRank = 0;
  for (const block of blocks) {
    if (!block.cacheable || block.tokens < model.minCacheTokens) continue;
    eligibleRank++;
    if (model.provider === 'anthropic' && eligibleRank > ANTHROPIC_MAX_CACHE_BREAKPOINTS) continue;
    ids.add(block.id);
  }
  return ids;
}

/**
 * Splits a prompt's cacheable blocks into "actually cacheable" (at/above the
 * model's `minCacheTokens` floor, and — for Anthropic — within the first 4
 * cache breakpoints in prompt order) and "marked cacheable but too small (or,
 * for Anthropic past the 4th breakpoint, too late) to cache" — the provider
 * would silently ignore the cache directive on the former two cases, so
 * treating their tokens as cache savings would overstate the `cachedCall`
 * estimate. Pure function of `blocks`/`model` — no I/O, easy to unit test in
 * isolation from the Svelte reactivity around it.
 */
export function evaluateCacheEligibility(blocks: PromptBlock[], model: ModelConfig): CacheEligibility {
  const cappedIds = cacheEligibleBlockIds(blocks, model);
  let eligibleCacheableTokens = 0;
  let belowFloorTokens = 0;
  const belowFloorBlockIds: string[] = [];
  let eligibleBlockCount = 0;

  for (const block of blocks) {
    if (!block.cacheable) continue;
    if (block.tokens >= model.minCacheTokens) {
      eligibleBlockCount++;
      // Past Anthropic's 4-breakpoint cap, a block still clears the floor
      // (so it still counts toward eligibleBlockCount, keeping the >4
      // warning accurate) but contributes no token savings — it would never
      // actually receive a cache_control marker in the export.
      if (cappedIds.has(block.id)) eligibleCacheableTokens += block.tokens;
    } else {
      belowFloorTokens += block.tokens;
      belowFloorBlockIds.push(block.id);
    }
  }

  const exceedsBreakpointLimit =
    model.provider === 'anthropic' && eligibleBlockCount > ANTHROPIC_MAX_CACHE_BREAKPOINTS;

  return { eligibleCacheableTokens, belowFloorTokens, belowFloorBlockIds, eligibleBlockCount, exceedsBreakpointLimit };
}

// ─── Exports (item 1) ───────────────────────────────────────────────────────

/**
 * Used by `exportAsOpenAIResponsesAPI` ONLY — `exportAsAnthropicMessages`
 * builds its own array-of-content-blocks shape directly (see that
 * function's doc comment) so per-block `cache_control` markers survive the
 * context-merge and turn-coalescing steps, which a joined string here
 * cannot carry.
 */
interface ExtractedPromptParts {
  /** Joined content of all `system` blocks — OpenAI's Responses API takes
   *  this as a single top-level string rather than a message in the array. */
  systemText: string;
  /** `user`/`assistant` turns only. `context` blocks are NOT emitted as
   *  their own turn — kept consistent with the Anthropic export's merge
   *  behavior (wrapped in [CONTEXT]...[/CONTEXT], merged into the FIRST
   *  user/assistant turn) rather than because this API is confirmed to need
   *  it: unlike the Anthropic Messages API (which documents and enforces
   *  strict user/assistant alternation starting with "user" — see
   *  `exportAsAnthropicMessages`/`anthropicExportRoleOrderWarning`), OpenAI's
   *  Chat Completions and Responses APIs are understood to NOT reject
   *  consecutive same-role entries; this project has no live network access
   *  to re-verify that against current docs as of this fix, so no
   *  role-order coalescing/rescue logic (item 2) has been added here — doing
   *  so would be guessing a constraint that may not exist. If that
   *  understanding turns out to be wrong, apply the same coalescing this
   *  file's Anthropic export uses. */
  messages: { role: 'user' | 'assistant'; content: string }[];
  /** Parsed tool definitions from every non-empty `tools` block, or
   *  `undefined` when there are none — so callers can omit the field
   *  entirely rather than exporting `tools: []`. */
  tools: unknown[] | undefined;
}

/** Defensive per-block JSON parse for `tools` blocks — a syntax error in one
 *  block must never crash the whole export. An unparseable block's raw text
 *  is passed through with an honest `_invalidJson` marker instead of being
 *  silently dropped or throwing. */
function parseToolsBlocks(blocks: PromptBlock[]): unknown[] | undefined {
  const toolsBlocks = blocks.filter((b) => b.role === 'tools' && b.content.trim().length > 0);
  if (toolsBlocks.length === 0) return undefined;

  return toolsBlocks.flatMap((b) => {
    try {
      const parsed: unknown = JSON.parse(b.content);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      return [{ _invalidJson: true, raw: b.content }];
    }
  });
}

function extractPromptParts(blocks: PromptBlock[]): ExtractedPromptParts {
  const systemText = blocks
    .filter((b) => b.role === 'system')
    .map((b) => b.content)
    .join('\n\n');

  const contextBlocks = blocks.filter((b) => b.role === 'context');
  const contextPrefix =
    contextBlocks.length > 0
      ? contextBlocks.map((b) => `[CONTEXT]\n${b.content}\n[/CONTEXT]`).join('\n\n') + '\n\n'
      : '';

  const turns = blocks.filter((b) => b.role === 'user' || b.role === 'assistant');
  const messages: { role: 'user' | 'assistant'; content: string }[] = turns.map((b, i) => ({
    role: b.role as 'user' | 'assistant',
    content: i === 0 ? contextPrefix + b.content : b.content,
  }));

  // Context blocks exist but there's no user/assistant turn to merge them
  // into — surface them as a single user turn rather than dropping them
  // from the export entirely.
  if (messages.length === 0 && contextPrefix) {
    messages.push({ role: 'user', content: contextPrefix.trimEnd() });
  }

  return { systemText, messages, tools: parseToolsBlocks(blocks) };
}

/** Anthropic's Messages API content-block shape for a text segment, with an
 *  optional cache-write marker. `cache_control` is only ever present (never
 *  `false`/`null`) so `JSON.stringify` omits the key entirely on
 *  non-cacheable segments instead of exporting a misleading falsy value. */
interface AnthropicTextBlock {
  type: 'text';
  text: string;
  cache_control?: { type: 'ephemeral' };
}

/**
 * True when `exportAsAnthropicMessages` would be forced to emit an
 * assistant-first `messages` array for `blocks` — invalid per the Messages
 * API, which requires the first message to have role `"user"`. This only
 * happens when the prompt's first `user`/`assistant` block is an `assistant`
 * block AND there is no `context` block available to rescue it into a
 * leading user turn (see that function's doc comment for the rescue rule).
 * `PromptApp.svelte` calls this to show a localized export warning instead
 * of silently handing the user an invalid payload.
 */
export function anthropicExportRoleOrderWarning(blocks: PromptBlock[]): boolean {
  const firstTurnRole = blocks.find((b) => b.role === 'user' || b.role === 'assistant')?.role;
  const hasContext = blocks.some((b) => b.role === 'context');
  return firstTurnRole === 'assistant' && !hasContext;
}

/**
 * Anthropic Messages API shape: `system` is a top-level ARRAY of text
 * content blocks — one per `system` block in prompt order — rather than a
 * message with role `"system"` (the real API rejects that) or a single
 * joined string (this file's earlier fix for that same bug). An array lets
 * each system block carry its own `cache_control` marker independently,
 * which a joined string cannot; that's the P0 fix this replaces the joined
 * string with — the caching panel already showed savings math for cacheable
 * blocks, but this export never actually encoded a cache marker anywhere, so
 * those savings were never realizable via copy-paste.
 *
 * `user`/`assistant` turns get the same array-of-content-blocks treatment.
 * `context` blocks are still not emitted as their own turn (the Messages API
 * requires strictly alternating user/assistant roles) — each one is merged
 * into the FIRST `user`/`assistant` turn as its own content segment (instead
 * of string-concatenation, so it can carry its own cache marker too), UNLESS
 * that first turn is `assistant` and rescuing is possible (see
 * `anthropicExportRoleOrderWarning`): in that case the context blocks become
 * a synthetic LEADING user turn instead, so the export still starts with a
 * user message. If the first turn is `assistant` and there's no context to
 * rescue it, the array is left assistant-first (not silently "fixed") and
 * `anthropicExportRoleOrderWarning` reports `true` so the caller can warn.
 *
 * Consecutive same-role turns (e.g. two `user` blocks in a row) are then
 * coalesced into one turn — the Messages API 400s on back-to-back same-role
 * entries — by concatenating their content-block arrays rather than
 * string-joining, so each part's own `cache_control` marker survives the
 * merge.
 *
 * Markers are capped at Anthropic's documented 4-breakpoint-per-request
 * limit via `cacheEligibleBlockIds`, the same function `evaluateCacheEligibility`
 * uses for the savings estimate above this export — so the panel and the
 * export can never disagree about which blocks are "really" cached.
 *
 * `tools`, when present, is passed through from the `tools` block(s) as-is
 * (see `parseToolsBlocks`) — NOT given a marker here: Anthropic caches tool
 * definitions via a `cache_control` on the LAST tool object, a distinct
 * mechanism from the text-content-block markers this function adds, and out
 * of scope for this fix.
 */
export function exportAsAnthropicMessages(blocks: PromptBlock[], model: ModelConfig): string {
  const cachedIds = cacheEligibleBlockIds(blocks, model);
  const toTextBlock = (block: PromptBlock, text: string): AnthropicTextBlock =>
    cachedIds.has(block.id) ? { type: 'text', text, cache_control: { type: 'ephemeral' } } : { type: 'text', text };
  const contextSegment = (ctx: PromptBlock): AnthropicTextBlock =>
    toTextBlock(ctx, `[CONTEXT]\n${ctx.content}\n[/CONTEXT]`);

  const system: AnthropicTextBlock[] = blocks
    .filter((b) => b.role === 'system')
    .map((b) => toTextBlock(b, b.content));

  const contextBlocks = blocks.filter((b) => b.role === 'context');
  const turnBlocks = blocks.filter((b) => b.role === 'user' || b.role === 'assistant');
  const contextRescueNeeded = turnBlocks[0]?.role === 'assistant' && contextBlocks.length > 0;

  type Turn = { role: 'user' | 'assistant'; content: AnthropicTextBlock[] };
  const rawTurns: Turn[] = turnBlocks.map((b, i) => {
    const segments: AnthropicTextBlock[] = [];
    // Context merges into the first turn UNLESS that turn is 'assistant' and
    // got rescued into its own leading turn instead (see below).
    if (i === 0 && contextBlocks.length > 0 && !contextRescueNeeded) {
      segments.push(...contextBlocks.map(contextSegment));
    }
    segments.push(toTextBlock(b, b.content));
    return { role: b.role as 'user' | 'assistant', content: segments };
  });

  if (contextRescueNeeded) {
    // First real turn is assistant — can't lead the API call with it.
    // Promote the context blocks (which would otherwise have merged into
    // that first turn) into their own leading user turn instead, so the
    // export still opens with role "user".
    rawTurns.unshift({ role: 'user', content: contextBlocks.map(contextSegment) });
  } else if (rawTurns.length === 0 && contextBlocks.length > 0) {
    // Context blocks exist but there's no user/assistant turn to merge them
    // into — surface them as a single user turn rather than dropping them.
    rawTurns.push({ role: 'user', content: contextBlocks.map(contextSegment) });
  }

  // Coalesce consecutive same-role turns (item 2) — appending content-block
  // arrays instead of string-joining preserves each part's own cache_control
  // marker as a separate segment within the merged turn.
  const messages: Turn[] = [];
  for (const turn of rawTurns) {
    const prev = messages[messages.length - 1];
    if (prev && prev.role === turn.role) {
      prev.content.push(...turn.content);
    } else {
      messages.push({ role: turn.role, content: [...turn.content] });
    }
  }

  const payload: Record<string, unknown> = { model: model.id, system, messages };
  const tools = parseToolsBlocks(blocks);
  if (tools) payload.tools = tools;
  return JSON.stringify(payload, null, 2);
}

/**
 * OpenAI Chat Completions shape — the tool's original/only export format,
 * kept byte-for-byte unchanged (`{ messages }`, `system` role included
 * inline in `messages` like the real API expects) when there's no `tools`
 * block, so existing exports stay valid; a `tools` field is only added when
 * the prompt actually has a tools block.
 */
export function exportAsOpenAIChatCompletions(blocks: PromptBlock[]): string {
  const messages = blocks
    .filter((b) => b.role !== 'tools')
    .map((b) => {
      let role: string = b.role;
      let content = b.content;

      if (role === 'context') {
        role = 'user';
        content = `[CONTEXT]\n${content}\n[/CONTEXT]`;
      }

      return { role, content };
    });

  const payload: Record<string, unknown> = { messages };
  const tools = parseToolsBlocks(blocks);
  if (tools) payload.tools = tools;
  return JSON.stringify(payload, null, 2);
}

/**
 * OpenAI Responses API shape: system content becomes the top-level
 * `instructions` string, and the conversation becomes the `input` array,
 * reusing `extractPromptParts`'s context-merge behavior for consistency with
 * the other exports — NOT because the Responses API is confirmed to require
 * strict user/assistant alternation the way Anthropic's Messages API does
 * (see `ExtractedPromptParts`'s doc comment for the honest uncertainty note
 * on that point, and item 2/FIX 9 of the audit this comment traces to).
 */
export function exportAsOpenAIResponsesAPI(blocks: PromptBlock[], model: ModelConfig): string {
  const { systemText, messages, tools } = extractPromptParts(blocks);
  const payload: Record<string, unknown> = { model: model.id, instructions: systemText, input: messages };
  if (tools) payload.tools = tools;
  return JSON.stringify(payload, null, 2);
}

/** Readable, human-facing dump of every block — NOT an API payload for any
 *  provider (no role remapping, no JSON structure a client library expects).
 *  Useful for pasting into a doc, chat window, or another tool by hand. */
export function exportAsPlainText(blocks: PromptBlock[]): string {
  return blocks.map(b => {
    return `<|${b.role}|>\n${b.content}\n`;
  }).join('\n');
}
