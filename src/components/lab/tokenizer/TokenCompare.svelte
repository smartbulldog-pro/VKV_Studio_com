<script lang="ts">
  import {
    tokenize,
    MODELS,
    MODEL_LIST,
    getModelInfo,
    type ModelId,
    type Token,
    type TokenResult,
    type Accuracy,
    type FallbackReason,
  } from '../../../lib/tokenizer/engine';
  import { t } from '../../../i18n/utils';

  let { text, lang = 'en' }: { text: string; lang?: 'en' | 'ru' } = $props();

  // Model A (primary)
  let modelA = $state<ModelId>(MODEL_LIST[0]);
  // Model B (comparison — default to second model)
  let modelB = $state<ModelId>(MODEL_LIST[3]);

  let resultA = $state<TokenResult | null>(null);
  let resultB = $state<TokenResult | null>(null);
  let isLoading = $state(false);
  // Distinguishes "a compare run failed" from "no text entered yet" — both
  // leave resultA/resultB null, so the template must not conflate them.
  let compareError = $state(false);

  // Race-condition guards: per-call request id + AbortController (mirrors
  // TokenizerApp.svelte). Plain (non-$state) locals — bookkeeping only.
  let latestRequestId = 0;
  let activeController: AbortController | null = null;

  // Check if model is API-based
  function isApiModel(modelId: ModelId): boolean {
    return getModelInfo(modelId).backend.type === 'api';
  }

  // Honest, reason-specific message for an 'approx' result — mirrors
  // TokenizerApp.svelte's getFallbackMessage(): a backend that's up but whose
  // provider rejected the request is not "backend offline", and an on-device
  // vocab load failure has nothing to do with the backend at all.
  function getFallbackMessage(reason: FallbackReason | undefined): string {
    switch (reason) {
      case 'provider':
        return t(lang, 'tokenizer.providerUnavailable');
      case 'onDeviceLoad':
        return t(lang, 'tokenizer.onDeviceTokenizerUnavailable');
      case 'closedWeight':
        return t(lang, 'tokenizer.closedWeightTokenizer');
      case 'network':
      default:
        return t(lang, 'tokenizer.backendOffline');
    }
  }

  // Get accuracy badge from a RESULT's explicit accuracy tier — never
  // inferred from tokens.length, so it can't contradict the engine. `forModel`
  // is needed for the 'verified' tier only — see TokenizerApp.svelte's
  // getAccuracyInfo for why (ModelInfo.verifiedViaSameFamilyModel).
  function getAccuracyBadge(
    accuracy: Accuracy,
    forModel: ModelId,
    fallbackReason?: FallbackReason
  ): { emoji: string; label: string; tooltip: string } {
    switch (accuracy) {
      case 'native':
        return {
          emoji: '🟢',
          label: t(lang, 'tokenizer.accuracyNative'),
          tooltip: t(lang, 'tokenizer.accuracyNativeTooltip'),
        };
      case 'verified':
        return {
          emoji: '🔵',
          label: t(lang, 'tokenizer.accuracyVerified'),
          tooltip: getModelInfo(forModel).verifiedViaSameFamilyModel
            ? t(lang, 'tokenizer.accuracyVerifiedSameFamilyTooltip')
            : t(lang, 'tokenizer.accuracyVerifiedTooltip'),
        };
      case 'approx':
        return {
          emoji: '🟡',
          label: t(lang, 'tokenizer.accuracyApprox'),
          tooltip: getFallbackMessage(fallbackReason),
        };
    }
  }

  // Debounce delay: 500ms for API, 250ms for local
  function getCompareDebounce(mA: ModelId, mB: ModelId): number {
    const hasApi = isApiModel(mA) || isApiModel(mB);
    return hasApi ? 500 : 250;
  }

  /**
   * Run both tokenize() calls for the current text/model pair. Guards
   * against races the same way TokenizerApp does: capture a request id
   * before awaiting, abort the previous in-flight pair, and discard the
   * result if a newer run has started by the time this one resolves.
   */
  async function runCompare(currentText: string, ma: ModelId, mb: ModelId) {
    const myRequestId = ++latestRequestId;
    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    compareError = false; // optimistic reset — this fresh run may succeed

    try {
      const [ra, rb] = await Promise.all([
        tokenize(currentText, ma, { signal: controller.signal }),
        tokenize(currentText, mb, { signal: controller.signal }),
      ]);

      if (myRequestId !== latestRequestId) return; // superseded — discard the stale result

      resultA = ra;
      resultB = rb;
      isLoading = false;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return; // intentionally cancelled
      if (myRequestId !== latestRequestId) return;
      isLoading = false;
      compareError = true;
    }
  }

  // Reactive: re-tokenize whenever text/models change
  let runTimer: ReturnType<typeof setTimeout>;
  $effect(() => {
    const currentText = text;
    const ma = modelA;
    const mb = modelB;

    clearTimeout(runTimer);

    if (!currentText.trim()) {
      // Cancel any in-flight request and invalidate it so a late resolution
      // can't repopulate resultA/resultB after the text was cleared.
      activeController?.abort();
      latestRequestId++;
      resultA = null;
      resultB = null;
      isLoading = false;
      compareError = false; // genuinely empty input, not a failed run
      return;
    }

    isLoading = true;
    const delay = getCompareDebounce(ma, mb);

    runTimer = setTimeout(() => {
      void runCompare(currentText, ma, mb);
    }, delay);

    return () => clearTimeout(runTimer);
  });

  // Derived: check if each panel has token breakdown
  let hasTokensA = $derived(resultA !== null && resultA.tokens.length > 0);
  let hasTokensB = $derived(resultB !== null && resultB.tokens.length > 0);
  // Per-token diff is only meaningful when BOTH sides expose a breakdown —
  // one empty side (verified/approx API count) used to make every token on
  // the other side look "different" by accident. Gate it explicitly instead.
  let canDiff = $derived(hasTokensA && hasTokensB);

  /**
   * Compute a simple diff:
   * Mark tokens in A and B that are "different" — meaning the text at that
   * position differs between the two streams.
   */
  function isDiff(tokens: Token[], otherTokens: Token[], idx: number): boolean {
    // Walk character positions to see if the boundary falls differently
    // We compare the running text at each token boundary
    let posA = 0;
    const boundsA: number[] = [];
    for (const tk of tokens) {
      posA += tk.text.length;
      boundsA.push(posA);
    }
    let posB = 0;
    const boundsB: number[] = [];
    for (const tk of otherTokens) {
      posB += tk.text.length;
      boundsB.push(posB);
    }
    const boundaryAtA = boundsA[idx];
    // If this boundary is not present in B → this token is a diff
    return !boundsB.includes(boundaryAtA);
  }
</script>

<div class="compare">
  <!-- Header row -->
  <div class="compare__header">
    <div class="compare__col-header">
      <label class="compare__col-label" for="compare-model-a">
        {t(lang, 'tokenizer.modelA')}
      </label>
      <select
        id="compare-model-a"
        class="compare__select"
        bind:value={modelA}
        aria-label={t(lang, 'tokenizer.modelA')}
      >
        {#each MODEL_LIST as mid (mid)}
          <option value={mid}>{MODELS[mid].name}</option>
        {/each}
      </select>
      {#if resultA}
        {@const badgeA = getAccuracyBadge(resultA.accuracy, modelA, resultA.fallbackReason)}
        <span
          class="compare__accuracy-badge compare__accuracy-badge--{resultA.accuracy}"
          title={badgeA.tooltip}
        >
          {badgeA.emoji}
          {badgeA.label}
        </span>
      {/if}
    </div>
    <div class="compare__divider" aria-hidden="true">vs</div>
    <div class="compare__col-header">
      <label class="compare__col-label" for="compare-model-b">
        {t(lang, 'tokenizer.modelB')}
      </label>
      <select
        id="compare-model-b"
        class="compare__select"
        bind:value={modelB}
        aria-label={t(lang, 'tokenizer.modelB')}
      >
        {#each MODEL_LIST as mid (mid)}
          <option value={mid}>{MODELS[mid].name}</option>
        {/each}
      </select>
      {#if resultB}
        {@const badgeB = getAccuracyBadge(resultB.accuracy, modelB, resultB.fallbackReason)}
        <span
          class="compare__accuracy-badge compare__accuracy-badge--{resultB.accuracy}"
          title={badgeB.tooltip}
        >
          {badgeB.emoji}
          {badgeB.label}
        </span>
      {/if}
    </div>
  </div>

  <!-- Stats row -->
  {#if resultA && resultB}
    <div class="compare__stats">
      <div class="compare__stat-col">
        <span class="compare__stat-num">{resultA.totalTokens}</span>
        <span class="compare__stat-lbl">{t(lang, 'tokenizer.tokensUnit')}</span>
        {#if resultA.density > 0}
          <span class="compare__stat-density">{resultA.density.toFixed(2)} char/tok</span>
        {/if}
      </div>
      <div class="compare__stat-center">
        {#if resultA.totalTokens !== resultB.totalTokens}
          <span class="compare__diff-badge compare__diff-badge--warn">
            Δ {Math.abs(resultA.totalTokens - resultB.totalTokens)}
          </span>
        {:else}
          <span class="compare__diff-badge compare__diff-badge--equal">
            = {t(lang, 'tokenizer.equal')}
          </span>
        {/if}
      </div>
      <div class="compare__stat-col compare__stat-col--right">
        <span class="compare__stat-num">{resultB.totalTokens}</span>
        <span class="compare__stat-lbl">{t(lang, 'tokenizer.tokensUnit')}</span>
        {#if resultB.density > 0}
          <span class="compare__stat-density">{resultB.density.toFixed(2)} char/tok</span>
        {/if}
      </div>
    </div>
  {/if}

  {#if resultA && resultB && !canDiff}
    <p class="compare__diff-unavailable">{t(lang, 'tokenizer.diffUnavailable')}</p>
  {/if}

  <!-- Token panels -->
  <div class="compare__panels">
    <!-- Panel A -->
    <div class="compare__panel" aria-label={t(lang, 'tokenizer.modelA')}>
      {#if isLoading}
        <span class="compare__loading">…</span>
      {:else if resultA}
        {#if hasTokensA}
          {#each resultA.tokens as token, i (token.id + '-a-' + i)}
            <span
              class="compare__token {canDiff && isDiff(resultA.tokens, resultB?.tokens ?? [], i)
                ? 'compare__token--diff'
                : ''}"
              style="--hue: {token.hue}"
              title="id: {token.id}">{token.text}</span
            >
          {/each}
        {:else}
          <!-- API model — no token breakdown -->
          <div class="compare__no-breakdown">
            <span class="compare__no-breakdown-count">
              {resultA.totalTokens.toLocaleString()}
              {t(lang, 'tokenizer.tokensUnit')}
            </span>
            <span class="compare__no-breakdown-msg">
              {t(lang, 'tokenizer.noBreakdown')}
            </span>
          </div>
        {/if}
      {:else if compareError}
        <span class="compare__empty compare__empty--error"
          >{t(lang, 'tokenizer.compareFailed')}</span
        >
      {:else}
        <span class="compare__empty">{t(lang, 'tokenizer.enterTextAbove')}</span>
      {/if}
    </div>

    <!-- Panel B -->
    <div class="compare__panel" aria-label={t(lang, 'tokenizer.modelB')}>
      {#if isLoading}
        <span class="compare__loading">…</span>
      {:else if resultB}
        {#if hasTokensB}
          {#each resultB.tokens as token, i (token.id + '-b-' + i)}
            <span
              class="compare__token {canDiff && isDiff(resultB.tokens, resultA?.tokens ?? [], i)
                ? 'compare__token--diff'
                : ''}"
              style="--hue: {token.hue}"
              title="id: {token.id}">{token.text}</span
            >
          {/each}
        {:else}
          <!-- API model — no token breakdown -->
          <div class="compare__no-breakdown">
            <span class="compare__no-breakdown-count">
              {resultB.totalTokens.toLocaleString()}
              {t(lang, 'tokenizer.tokensUnit')}
            </span>
            <span class="compare__no-breakdown-msg">
              {t(lang, 'tokenizer.noBreakdown')}
            </span>
          </div>
        {/if}
      {:else if compareError}
        <span class="compare__empty compare__empty--error"
          >{t(lang, 'tokenizer.compareFailed')}</span
        >
      {:else}
        <span class="compare__empty">{t(lang, 'tokenizer.enterTextAbove')}</span>
      {/if}
    </div>
  </div>
</div>

<style>
  .compare {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }

  /* Header */
  .compare__header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: var(--space-4);
  }

  .compare__col-header {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .compare__col-label {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
  }

  .compare__select {
    background: var(--bg-slate);
    color: var(--text-primary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-3);
    font-family: var(--font-sans);
    font-size: var(--text-xs);
    outline: none;
    cursor: pointer;
    transition: var(--transition-colors);
  }

  .compare__select:hover,
  .compare__select:focus {
    border-color: var(--accent-green-300);
  }

  .compare__accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--text-muted);
    cursor: help;
    margin-top: var(--space-1);
    padding: 1px var(--space-2);
    border-radius: var(--radius-full);
    width: fit-content;
  }

  /* Tier coloring — consistent with .tokenizer__accuracy-badge--* in tokenizer.css */
  .compare__accuracy-badge--native {
    background: hsla(155, 70%, 50%, 0.1);
    border: 1px solid hsla(155, 70%, 50%, 0.3);
    color: var(--accent-green-300);
  }

  .compare__accuracy-badge--verified {
    background: hsla(225, 65%, 55%, 0.12);
    border: 1px solid hsla(225, 65%, 55%, 0.3);
    color: var(--accent-blue-200, hsl(225, 65%, 70%));
  }

  .compare__accuracy-badge--approx {
    background: hsla(40, 90%, 55%, 0.1);
    border: 1px solid hsla(40, 90%, 55%, 0.3);
    color: hsl(40, 90%, 65%);
  }

  .compare__diff-unavailable {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--text-muted);
    font-style: italic;
    text-align: center;
  }

  .compare__divider {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--text-muted);
    padding: 0 var(--space-2);
    text-align: center;
  }

  /* Stats */
  .compare__stats {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: var(--space-2);
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
  }

  .compare__stat-col {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .compare__stat-col--right {
    align-items: flex-end;
  }

  .compare__stat-num {
    font-family: var(--font-mono);
    font-size: var(--text-h4);
    font-weight: var(--weight-bold);
    color: var(--text-primary);
  }

  .compare__stat-lbl {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
  }

  .compare__stat-density {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--text-ghost);
  }

  .compare__stat-center {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .compare__diff-badge {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-full);
    letter-spacing: var(--tracking-wide);
  }

  .compare__diff-badge--warn {
    background: hsla(40, 90%, 55%, 0.15);
    border: 1px solid hsla(40, 90%, 55%, 0.4);
    color: hsl(40, 90%, 65%);
  }

  .compare__diff-badge--equal {
    background: hsla(155, 70%, 50%, 0.12);
    border: 1px solid hsla(155, 70%, 50%, 0.35);
    color: var(--accent-green-300);
  }

  /* Panels */
  .compare__panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }

  @media (max-width: 640px) {
    .compare__panels {
      grid-template-columns: 1fr;
    }

    /* Prevent iOS Safari auto-zoom on focus (requires >=16px font-size) */
    .compare__select {
      font-size: 16px;
    }
  }

  .compare__panel {
    background: var(--bg-slate);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    min-height: 120px;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    line-height: var(--leading-relaxed);
    word-wrap: break-word;
    white-space: pre-wrap;
    overflow-y: auto;
    max-height: 300px;
  }

  .compare__token {
    display: inline-block;
    padding: 0 1px;
    margin: 0 1px;
    border-radius: 2px;
    background-color: hsla(var(--hue), 70%, 50%, 0.18);
    color: var(--text-primary);
    border-bottom: 2px solid hsla(var(--hue), 70%, 50%, 0.6);
    transition: background-color var(--duration-fast) var(--ease-out);

    @media (prefers-reduced-motion: no-preference) {
      opacity: 0;
      animation: compare-token-in 200ms var(--ease-out) forwards;
    }
  }

  @media (prefers-reduced-motion: no-preference) {
    @keyframes compare-token-in {
      from {
        opacity: 0;
        transform: translateY(4px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .compare__token {
      opacity: 1 !important;
      animation: none !important;
    }
  }

  /* Diff highlight */
  .compare__token--diff {
    background-color: hsla(40, 90%, 55%, 0.22) !important;
    border-bottom-color: hsl(40, 90%, 65%) !important;
    outline: 1px dashed hsla(40, 90%, 55%, 0.6);
  }

  .compare__loading {
    color: var(--text-muted);
    font-family: var(--font-mono);
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 0.4;
    }
    50% {
      opacity: 1;
    }
  }

  .compare__empty {
    color: var(--text-ghost);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-style: italic;
  }

  .compare__empty--error {
    color: var(--color-error);
  }

  /* No breakdown message for API models */
  .compare__no-breakdown {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    padding: var(--space-6) var(--space-4);
    text-align: center;
    min-height: 80px;
  }

  .compare__no-breakdown-count {
    font-family: var(--font-mono);
    font-size: var(--text-h4);
    font-weight: var(--weight-bold);
    color: var(--accent-green-300);
  }

  .compare__no-breakdown-msg {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-muted);
    font-style: italic;
    max-width: 280px;
    line-height: var(--leading-relaxed);
  }
</style>
