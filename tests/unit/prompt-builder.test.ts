/**
 * Lab Quest Phase B — pure-function regression tests for lib/prompt/builder.ts.
 * =============================================================================
 * Covers the audit findings fixed in this phase: the cache floor (item 2),
 * batch-API discount math (item 4), provider-correct exports (item 1), and
 * honest chat-wrapping-overhead tier selection (item 3). All pure functions —
 * no DOM, no network, no Svelte runtime.
 */
import { describe, it, expect } from 'vitest';
import {
  type PromptBlock,
  type ModelConfig,
  evaluateCacheEligibility,
  calculateCachedCost,
  chatWrappingOverhead,
  exportAsAnthropicMessages,
  anthropicExportRoleOrderWarning,
  exportAsOpenAIChatCompletions,
  exportAsOpenAIResponsesAPI,
} from '@/lib/prompt/builder';
import {
  API_CHAT_OVERHEAD_ESTIMATE_TOKENS,
  OPENAI_CHATML_TOKENS_PER_MESSAGE,
  OPENAI_CHATML_REPLY_PRIMING_TOKENS,
} from '@/lib/tokenizer/engine';

// ─── Fixtures ────────────────────────────────────────────────────────────────

function makeModel(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    id: 'claude-sonnet-5',
    name: 'Claude Sonnet 5 (test fixture)',
    contextWindow: 1_000_000,
    inputPrice: 2.0,
    outputPrice: 10.0,
    supportsCaching: true,
    cacheWriteMultiplier: 1.25,
    cacheReadMultiplier: 0.1,
    provider: 'anthropic',
    minCacheTokens: 1024,
    batchDiscount: 0.5,
    reasoningCapable: true,
    ...overrides,
  };
}

function makeBlock(overrides: Partial<PromptBlock> = {}): PromptBlock {
  return {
    id: crypto.randomUUID(),
    role: 'user',
    content: '',
    tokens: 0,
    collapsed: false,
    cacheable: false,
    ...overrides,
  };
}

// ─── evaluateCacheEligibility — cache floor (item 2) ─────────────────────────

describe('evaluateCacheEligibility — cache floor', () => {
  it('excludes a cacheable block below minCacheTokens from eligible tokens', () => {
    const model = makeModel({ minCacheTokens: 1024 });
    const blocks = [
      makeBlock({ role: 'system', cacheable: true, tokens: 2000 }), // above floor
      makeBlock({ role: 'context', cacheable: true, tokens: 500 }), // below floor
    ];
    const result = evaluateCacheEligibility(blocks, model);
    expect(result.eligibleCacheableTokens).toBe(2000);
    expect(result.belowFloorTokens).toBe(500);
    expect(result.belowFloorBlockIds).toEqual([blocks[1].id]);
    expect(result.eligibleBlockCount).toBe(1);
  });

  it('a block exactly AT the floor counts as eligible (>=, not >)', () => {
    const model = makeModel({ minCacheTokens: 1024 });
    const blocks = [makeBlock({ role: 'system', cacheable: true, tokens: 1024 })];
    const result = evaluateCacheEligibility(blocks, model);
    expect(result.eligibleCacheableTokens).toBe(1024);
    expect(result.belowFloorTokens).toBe(0);
  });

  it('non-cacheable blocks never contribute to either bucket, regardless of size', () => {
    const model = makeModel({ minCacheTokens: 1024 });
    const blocks = [makeBlock({ role: 'user', cacheable: false, tokens: 50_000 })];
    const result = evaluateCacheEligibility(blocks, model);
    expect(result.eligibleCacheableTokens).toBe(0);
    expect(result.belowFloorTokens).toBe(0);
    expect(result.eligibleBlockCount).toBe(0);
  });

  it('flags exceedsBreakpointLimit only past Anthropic\'s 4-breakpoint limit, and only for anthropic', () => {
    const anthropic = makeModel({ provider: 'anthropic', minCacheTokens: 100 });
    const fourEligible = Array.from({ length: 4 }, () =>
      makeBlock({ role: 'context', cacheable: true, tokens: 200 })
    );
    expect(evaluateCacheEligibility(fourEligible, anthropic).exceedsBreakpointLimit).toBe(false);

    const fiveEligible = Array.from({ length: 5 }, () =>
      makeBlock({ role: 'context', cacheable: true, tokens: 200 })
    );
    const overLimit = evaluateCacheEligibility(fiveEligible, anthropic);
    expect(overLimit.eligibleBlockCount).toBe(5);
    expect(overLimit.exceedsBreakpointLimit).toBe(true);

    // Same 5-eligible-block shape, but a non-Anthropic provider: no documented
    // 4-breakpoint limit for it here, so this must stay false rather than
    // borrowing Anthropic's number.
    const openai = makeModel({ provider: 'openai', minCacheTokens: 100 });
    expect(evaluateCacheEligibility(fiveEligible, openai).exceedsBreakpointLimit).toBe(false);
  });

  // ── FIX 1b: the savings estimate must agree with the export's marker cap ──
  it('caps eligibleCacheableTokens at the first 4 eligible blocks for anthropic — matches the export cache_control cap', () => {
    const anthropic = makeModel({ provider: 'anthropic', minCacheTokens: 100 });
    const fiveEligible = Array.from({ length: 5 }, () =>
      makeBlock({ role: 'context', cacheable: true, tokens: 200 })
    );
    const result = evaluateCacheEligibility(fiveEligible, anthropic);
    expect(result.eligibleBlockCount).toBe(5); // the >4 warning still fires off this count
    expect(result.eligibleCacheableTokens).toBe(800); // only the first 4 × 200 tokens

    // Same 5-eligible-block shape, non-Anthropic provider: no cap.
    const openai = makeModel({ provider: 'openai', minCacheTokens: 100 });
    const openaiResult = evaluateCacheEligibility(fiveEligible, openai);
    expect(openaiResult.eligibleCacheableTokens).toBe(1000); // full sum, no breakpoint ceiling
  });
});

// ─── calculateCachedCost — batch discount math (item 4) ──────────────────────

describe('calculateCachedCost — batch discount', () => {
  it('defaults to no discount when batchMultiplier is omitted', () => {
    const model = makeModel();
    const withDefault = calculateCachedCost(2000, 100, 500, model);
    const withExplicitOne = calculateCachedCost(2000, 100, 500, model, 1);
    expect(withDefault).toEqual(withExplicitOne);
  });

  it('halves both firstCall and cachedCall at a 0.5 batch multiplier', () => {
    const model = makeModel();
    const full = calculateCachedCost(2000, 100, 500, model, 1);
    const batched = calculateCachedCost(2000, 100, 500, model, 0.5);
    expect(batched.firstCall).toBeCloseTo(full.firstCall * 0.5, 10);
    expect(batched.cachedCall).toBeCloseTo(full.cachedCall * 0.5, 10);
  });

  it('stacks multiplicatively with the cache discount, not additively', () => {
    // write = cacheableTokens/1e6 * inputPrice * cacheWriteMultiplier
    // read  = cacheableTokens/1e6 * inputPrice * cacheReadMultiplier
    const model = makeModel({ inputPrice: 2.0, cacheWriteMultiplier: 1.25, cacheReadMultiplier: 0.1 });
    const cacheableTokens = 10_000;
    const freshTokens = 0;
    const outputTokens = 0;
    const result = calculateCachedCost(cacheableTokens, freshTokens, outputTokens, model, 0.5);
    const expectedWrite = (cacheableTokens / 1_000_000) * model.inputPrice * model.cacheWriteMultiplier;
    const expectedRead = (cacheableTokens / 1_000_000) * model.inputPrice * model.cacheReadMultiplier;
    expect(result.firstCall).toBeCloseTo(expectedWrite * 0.5, 10);
    expect(result.cachedCall).toBeCloseTo(expectedRead * 0.5, 10);
  });

  it('applies the batch multiplier to the flat cost of a non-caching model too', () => {
    const model = makeModel({ supportsCaching: false, inputPrice: 1.0, outputPrice: 1.0 });
    const result = calculateCachedCost(1_000_000, 0, 0, model, 0.5);
    expect(result.firstCall).toBeCloseTo(0.5, 10);
    expect(result.cachedCall).toBe(result.firstCall);
  });

  it('savingsPct is unaffected by the batch multiplier (a ratio, scales out)', () => {
    const model = makeModel();
    const full = calculateCachedCost(5000, 200, 300, model, 1);
    const batched = calculateCachedCost(5000, 200, 300, model, 0.5);
    expect(batched.savingsPct).toBeCloseTo(full.savingsPct, 10);
  });
});

// ─── Exports (item 1) ─────────────────────────────────────────────────────────

describe('exportAsAnthropicMessages', () => {
  it('puts system content top-level as an array of text blocks, never as a message with role "system"', () => {
    const model = makeModel({ id: 'claude-sonnet-5', minCacheTokens: 0 });
    const blocks = [
      makeBlock({ role: 'system', content: 'You are helpful.' }),
      makeBlock({ role: 'user', content: 'Hi there' }),
      makeBlock({ role: 'assistant', content: 'Hello!' }),
    ];
    const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
    expect(parsed.system).toEqual([{ type: 'text', text: 'You are helpful.' }]);
    expect(parsed.model).toBe('claude-sonnet-5');
    expect(Array.isArray(parsed.messages)).toBe(true);
    expect(parsed.messages.some((m: { role: string }) => m.role === 'system')).toBe(false);
    expect(parsed.messages).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'Hi there' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Hello!' }] },
    ]);
  });

  it('merges context blocks into the first user/assistant turn as separate content segments instead of emitting a separate turn', () => {
    const model = makeModel();
    const blocks = [
      makeBlock({ role: 'system', content: 'sys' }),
      makeBlock({ role: 'context', content: 'retrieved docs here' }),
      makeBlock({ role: 'user', content: 'question' }),
      makeBlock({ role: 'assistant', content: 'answer' }),
    ];
    const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
    // Exactly 2 turns (user, assistant) — context did NOT become its own turn,
    // which would otherwise produce two consecutive 'user' roles.
    expect(parsed.messages).toHaveLength(2);
    expect(parsed.messages[0].role).toBe('user');
    expect(parsed.messages[0].content).toHaveLength(2);
    expect(parsed.messages[0].content[0].text).toContain('[CONTEXT]');
    expect(parsed.messages[0].content[0].text).toContain('retrieved docs here');
    expect(parsed.messages[0].content[1].text).toBe('question');
    expect(parsed.messages[1]).toEqual({ role: 'assistant', content: [{ type: 'text', text: 'answer' }] });
  });

  it('passes through valid JSON tools content as a top-level tools array', () => {
    const model = makeModel();
    const toolDefs = [{ name: 'get_weather', input_schema: { type: 'object' } }];
    const blocks = [
      makeBlock({ role: 'system', content: 'sys' }),
      makeBlock({ role: 'tools', content: JSON.stringify(toolDefs) }),
      makeBlock({ role: 'user', content: 'q' }),
    ];
    const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
    expect(parsed.tools).toEqual(toolDefs);
    // Tools block must not leak into messages[] as a chat turn.
    expect(parsed.messages.some((m: { role: string }) => m.role === 'tools')).toBe(false);
  });

  it('never crashes on invalid JSON in a tools block — falls back to an honest raw marker', () => {
    const model = makeModel();
    const blocks = [makeBlock({ role: 'tools', content: '{not valid json' })];
    expect(() => exportAsAnthropicMessages(blocks, model)).not.toThrow();
    const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
    expect(parsed.tools).toEqual([{ _invalidJson: true, raw: '{not valid json' }]);
  });

  it('omits the tools field entirely when there is no tools block', () => {
    const model = makeModel();
    const blocks = [makeBlock({ role: 'user', content: 'hi' })];
    const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
    expect('tools' in parsed).toBe(false);
  });

  // ── FIX 1a: real cache_control markers ────────────────────────────────────
  describe('cache_control markers', () => {
    it('marks a cacheable system block with cache_control', () => {
      const model = makeModel({ minCacheTokens: 0 });
      const blocks = [makeBlock({ role: 'system', content: 'sys A', cacheable: true, tokens: 10 })];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect(parsed.system).toEqual([
        { type: 'text', text: 'sys A', cache_control: { type: 'ephemeral' } },
      ]);
    });

    it('leaves a non-cacheable system block plain — no cache_control key at all', () => {
      const model = makeModel({ minCacheTokens: 0 });
      const blocks = [makeBlock({ role: 'system', content: 'sys', cacheable: false, tokens: 10 })];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect(parsed.system).toEqual([{ type: 'text', text: 'sys' }]);
      expect('cache_control' in parsed.system[0]).toBe(false);
    });

    it('a cacheable block below minCacheTokens gets no marker — matches evaluateCacheEligibility\'s floor', () => {
      const model = makeModel({ minCacheTokens: 1024 });
      const blocks = [makeBlock({ role: 'system', content: 'small', cacheable: true, tokens: 5 })];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect('cache_control' in parsed.system[0]).toBe(false);
    });

    it('marks a cacheable user/assistant block\'s own content segment', () => {
      const model = makeModel({ minCacheTokens: 0 });
      const blocks = [
        makeBlock({ role: 'user', content: 'few-shot example', cacheable: true, tokens: 10 }),
      ];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect(parsed.messages[0].content).toEqual([
        { type: 'text', text: 'few-shot example', cache_control: { type: 'ephemeral' } },
      ]);
    });

    it('caps markers at the first 4 cacheable blocks in prompt order — a 5th cacheable block is exported plain', () => {
      const model = makeModel({ provider: 'anthropic', minCacheTokens: 0 });
      const blocks = [
        makeBlock({ role: 'system', content: 'a', cacheable: true, tokens: 10 }),
        makeBlock({ role: 'context', content: 'b', cacheable: true, tokens: 10 }),
        makeBlock({ role: 'context', content: 'c', cacheable: true, tokens: 10 }),
        makeBlock({ role: 'context', content: 'd', cacheable: true, tokens: 10 }),
        makeBlock({ role: 'user', content: 'e', cacheable: true, tokens: 10 }),
      ];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      // 'a' (system) is the 1st cacheable block in prompt order.
      expect(parsed.system[0].cache_control).toEqual({ type: 'ephemeral' });
      // The single user turn's content is [b, c, d, e] (context merged + the
      // user block itself) — b/c/d are the 2nd/3rd/4th cacheable blocks and
      // get markers; e is the 5th and is exported plain.
      const segments = parsed.messages[0].content;
      expect(segments).toHaveLength(4);
      expect(segments[0].cache_control).toEqual({ type: 'ephemeral' }); // b
      expect(segments[1].cache_control).toEqual({ type: 'ephemeral' }); // c
      expect(segments[2].cache_control).toEqual({ type: 'ephemeral' }); // d
      expect('cache_control' in segments[3]).toBe(false); // e — 5th, past the cap
    });
  });

  // ── FIX 2: alternation — coalescing + user-first guarantee ────────────────
  describe('role alternation', () => {
    it('coalesces two consecutive user blocks into one turn with two content blocks', () => {
      const model = makeModel();
      const blocks = [
        makeBlock({ role: 'system', content: 'sys' }),
        makeBlock({ role: 'user', content: 'first' }),
        makeBlock({ role: 'user', content: 'second' }),
      ];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect(parsed.messages).toHaveLength(1);
      expect(parsed.messages[0].role).toBe('user');
      expect(parsed.messages[0].content).toEqual([
        { type: 'text', text: 'first' },
        { type: 'text', text: 'second' },
      ]);
    });

    it('coalesces three consecutive assistant blocks into one turn preserving each cache marker', () => {
      const model = makeModel({ minCacheTokens: 0 });
      const blocks = [
        makeBlock({ role: 'user', content: 'q' }),
        makeBlock({ role: 'assistant', content: 'a1', cacheable: true, tokens: 10 }),
        makeBlock({ role: 'assistant', content: 'a2', cacheable: false }),
        makeBlock({ role: 'assistant', content: 'a3', cacheable: true, tokens: 10 }),
      ];
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, model));
      expect(parsed.messages).toHaveLength(2);
      expect(parsed.messages[1].role).toBe('assistant');
      expect(parsed.messages[1].content).toEqual([
        { type: 'text', text: 'a1', cache_control: { type: 'ephemeral' } },
        { type: 'text', text: 'a2' },
        { type: 'text', text: 'a3', cache_control: { type: 'ephemeral' } },
      ]);
    });

    it('does not flag a normal user-first prompt', () => {
      const blocks = [makeBlock({ role: 'user', content: 'hi' })];
      expect(anthropicExportRoleOrderWarning(blocks)).toBe(false);
    });

    it('flags an assistant-first prompt with no context block to rescue it, and leaves the export as-is rather than silently "fixing" it', () => {
      const blocks = [makeBlock({ role: 'assistant', content: 'hi' })];
      expect(anthropicExportRoleOrderWarning(blocks)).toBe(true);
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, makeModel()));
      expect(parsed.messages[0].role).toBe('assistant');
    });

    it('rescues an assistant-first prompt into a leading user turn when a context block exists', () => {
      const blocks = [
        makeBlock({ role: 'context', content: 'background' }),
        makeBlock({ role: 'assistant', content: 'hi' }),
      ];
      expect(anthropicExportRoleOrderWarning(blocks)).toBe(false);
      const parsed = JSON.parse(exportAsAnthropicMessages(blocks, makeModel()));
      expect(parsed.messages).toHaveLength(2);
      expect(parsed.messages[0].role).toBe('user');
      expect(parsed.messages[0].content[0].text).toContain('background');
      expect(parsed.messages[1]).toEqual({ role: 'assistant', content: [{ type: 'text', text: 'hi' }] });
    });
  });
});

describe('exportAsOpenAIChatCompletions', () => {
  it('stays byte-identical to the pre-Phase-B shape when there is no tools block', () => {
    const blocks = [
      makeBlock({ role: 'system', content: 'sys' }),
      makeBlock({ role: 'user', content: 'hi' }),
    ];
    const parsed = JSON.parse(exportAsOpenAIChatCompletions(blocks));
    expect(Object.keys(parsed)).toEqual(['messages']);
    expect(parsed.messages).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ]);
  });

  it('adds a top-level tools field, and excludes tools blocks from messages[]', () => {
    const toolDefs = [{ type: 'function', function: { name: 'lookup' } }];
    const blocks = [
      makeBlock({ role: 'system', content: 'sys' }),
      makeBlock({ role: 'tools', content: JSON.stringify(toolDefs) }),
      makeBlock({ role: 'user', content: 'hi' }),
    ];
    const parsed = JSON.parse(exportAsOpenAIChatCompletions(blocks));
    expect(parsed.tools).toEqual(toolDefs);
    expect(parsed.messages).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ]);
  });
});

describe('exportAsOpenAIResponsesAPI', () => {
  it('uses instructions/input, not system/messages', () => {
    const model = makeModel({ id: 'gpt-5.5' });
    const blocks = [
      makeBlock({ role: 'system', content: 'sys prompt' }),
      makeBlock({ role: 'user', content: 'hi' }),
    ];
    const parsed = JSON.parse(exportAsOpenAIResponsesAPI(blocks, model));
    expect(parsed.model).toBe('gpt-5.5');
    expect(parsed.instructions).toBe('sys prompt');
    expect(parsed.input).toEqual([{ role: 'user', content: 'hi' }]);
    expect('system' in parsed).toBe(false);
    expect('messages' in parsed).toBe(false);
  });

  it('shares the same context-merge and tools passthrough behavior as the Anthropic export', () => {
    const model = makeModel({ id: 'gpt-5.5' });
    const toolDefs = [{ name: 'search' }];
    const blocks = [
      makeBlock({ role: 'context', content: 'ctx' }),
      makeBlock({ role: 'tools', content: JSON.stringify(toolDefs) }),
      makeBlock({ role: 'user', content: 'q' }),
    ];
    const parsed = JSON.parse(exportAsOpenAIResponsesAPI(blocks, model));
    expect(parsed.input).toHaveLength(1);
    expect(parsed.input[0].content).toContain('[CONTEXT]');
    expect(parsed.tools).toEqual(toolDefs);
  });
});

// ─── chatWrappingOverhead — honesty tier selection (item 3) ──────────────────

describe('chatWrappingOverhead', () => {
  it('returns 0 overhead for zero messages, tagged documented (no guess made)', () => {
    const result = chatWrappingOverhead(0, makeModel());
    expect(result).toEqual({ overhead: 0, exactness: 'documented' });
  });

  it('uses the documented OpenAI ChatML formula for a tiktoken-backed model', () => {
    // gpt-5.5 is tiktoken-backed in tokenizer/engine.ts.
    const model = makeModel({ id: 'gpt-5.5', provider: 'openai' });
    const messageCount = 4;
    const result = chatWrappingOverhead(messageCount, model);
    expect(result.exactness).toBe('documented');
    expect(result.overhead).toBe(
      messageCount * OPENAI_CHATML_TOKENS_PER_MESSAGE + OPENAI_CHATML_REPLY_PRIMING_TOKENS
    );
  });

  it('falls back to a flat estimated tier for an API-backed (Anthropic) model — never "documented"', () => {
    const model = makeModel({ id: 'claude-sonnet-5', provider: 'anthropic' });
    const messageCount = 3;
    const result = chatWrappingOverhead(messageCount, model);
    expect(result.exactness).toBe('estimated');
    expect(result.overhead).toBe(messageCount * API_CHAT_OVERHEAD_ESTIMATE_TOKENS);
  });

  it('falls back to the same estimated tier for a self-hosted transformers-backed model', () => {
    const model = makeModel({ id: 'gemma-4-e2b', provider: 'local' });
    const result = chatWrappingOverhead(2, model);
    expect(result.exactness).toBe('estimated');
  });

  it('an unknown model id degrades to the estimated tier rather than throwing', () => {
    const model = makeModel({ id: 'not-a-real-model-id' });
    expect(() => chatWrappingOverhead(2, model)).not.toThrow();
    expect(chatWrappingOverhead(2, model).exactness).toBe('estimated');
  });
});
