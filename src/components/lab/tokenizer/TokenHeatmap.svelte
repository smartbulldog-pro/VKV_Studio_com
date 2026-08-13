<script lang="ts">
  import type { Token } from '../../../lib/tokenizer/engine';
  import { placeTooltip, pointerXY } from '../../../lib/tokenizer/tooltip-position';
  import { t } from '../../../i18n/utils';

  let { tokens, lang = 'en' }: { tokens: Token[]; lang?: 'en' | 'ru' } = $props();

  // Tooltip state
  let tooltip = $state<{
    visible: boolean;
    x: number;
    y: number;
    token: Token | null;
  }>({ visible: false, x: 0, y: 0, token: null });

  /** Map token byte-length to a heatmap color: short = dark teal, long = bright cyan */
  function heatColor(text: string): string {
    const len = Math.min(text.length, 12); // clamp at 12 chars
    // 0..12 → lightness 18%..65%
    const lightness = 18 + (len / 12) * 47;
    // 0..12 → saturation 30%..85%
    const sat = 30 + (len / 12) * 55;
    return `hsl(185, ${sat.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
  }

  /** Text color: light text on dark backgrounds, dark on bright */
  function textColor(text: string): string {
    const len = Math.min(text.length, 12);
    const lightness = 18 + (len / 12) * 47;
    return lightness > 42 ? 'hsl(220, 20%, 6%)' : 'hsl(210, 20%, 90%)';
  }

  function showTooltip(e: MouseEvent | TouchEvent, token: Token) {
    const pointer = pointerXY(e);
    const { x, y } = placeTooltip(pointer.x, pointer.y);
    tooltip = { visible: true, x, y, token };
  }

  function hideTooltip() {
    tooltip = { ...tooltip, visible: false };
  }
</script>

<!-- Dismiss tooltip on viewport tap -->
<svelte:window onclick={hideTooltip} />

<div class="heatmap" aria-label={t(lang, 'tokenizer.heatmapLabel')}>
  {#each tokens as token, i (token.id + '-' + i)}
    <!-- svelte-ignore a11y_mouse_events_have_key_events -->
    <span
      class="heatmap__token"
      style="
        --heat-bg: {heatColor(token.text)};
        --heat-text: {textColor(token.text)};
        animation-delay: {Math.min(i, 60) * 20}ms;
      "
      role="button"
      tabindex="0"
      aria-label={t(lang, 'tokenizer.tokenAriaLabel').replace('{id}', String(token.id)).replace('{text}', token.text)}
      onmousemove={(e) => showTooltip(e, token)}
      onmouseleave={hideTooltip}
      onclick={(e) => {
        e.stopPropagation();
        showTooltip(e, token);
      }}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          showTooltip(e, token);
        }
      }}
    >{token.text}</span>
  {/each}
</div>

{#if tooltip.visible && tooltip.token}
  <div
    class="heatmap__tooltip"
    style="left: {tooltip.x}px; top: {tooltip.y}px;"
    role="tooltip"
  >
    <div class="heatmap__tooltip-row">
      <span class="heatmap__tooltip-key">id</span>
      <span class="heatmap__tooltip-val">{tooltip.token.id}</span>
    </div>
    <div class="heatmap__tooltip-row">
      <span class="heatmap__tooltip-key">text</span>
      <span class="heatmap__tooltip-val heatmap__tooltip-text">"{tooltip.token.text.replace(/\n/g, '↵').replace(/ /g, '·')}"</span>
    </div>
    {#if tooltip.token.partial}
      <div class="heatmap__tooltip-row">
        <span class="heatmap__tooltip-key">bytes</span>
        <span class="heatmap__tooltip-val heatmap__tooltip-val--partial" title={t(lang, 'tokenizer.partialByteSequenceTooltip')}>
          ⚠ {t(lang, 'tokenizer.partialByteSequence')}
        </span>
      </div>
    {:else}
      <div class="heatmap__tooltip-row">
        <span class="heatmap__tooltip-key">bytes</span>
        <span class="heatmap__tooltip-val">[{tooltip.token.bytes.join(', ')}]</span>
      </div>
      <div class="heatmap__tooltip-row">
        <span class="heatmap__tooltip-key">len</span>
        <span class="heatmap__tooltip-val">{tooltip.token.bytes.length}B</span>
      </div>
    {/if}
  </div>
{/if}

<style>
  .heatmap {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: var(--leading-relaxed);
    word-wrap: break-word;
    white-space: pre-wrap;
  }

  .heatmap__token {
    display: inline;
    background-color: var(--heat-bg);
    color: var(--heat-text);
    padding: 0.5px 0;
    cursor: pointer;
    user-select: none;
    transition: filter var(--duration-fast) var(--ease-out),
                outline var(--duration-fast) var(--ease-out),
                transform var(--duration-fast) var(--ease-out);

    /* animation-delay is set inline as `min(i, 60) * 20ms` above — without
       the cap, large pasted input (~16k tokens) queued the last token's
       entrance minutes out (silent background jank, mirrors the same fix
       in tokenizer.css's .token). */
    @media (prefers-reduced-motion: no-preference) {
      opacity: 0;
      animation: heatmap-token-in 250ms var(--ease-out) forwards;
    }
  }

  @media (prefers-reduced-motion: no-preference) {
    @keyframes heatmap-token-in {
      from { opacity: 0; filter: blur(3px); }
      to   { opacity: 1; filter: blur(0); }
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .heatmap__token {
      opacity: 1 !important;
      animation: none !important;
    }
  }

  .heatmap__token:hover,
  .heatmap__token:focus {
    filter: brightness(1.3);
    outline: 1px solid hsla(185, 80%, 60%, 0.7);
    outline-offset: 1px;
    z-index: 1;
    position: relative;
  }

  .heatmap__token:active {
    transform: scale(0.95);
  }

  /* Tooltip */
  .heatmap__tooltip {
    position: fixed;
    z-index: 9999;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg), 0 0 16px hsla(185, 70%, 50%, 0.2);
    padding: var(--space-3) var(--space-4);
    pointer-events: none;
    min-width: 180px;
    /* Keep in sync with TOOLTIP_MAX_W in lib/tokenizer/tooltip-position.ts. */
    max-width: min(280px, calc(100vw - 24px));
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
  }

  .heatmap__tooltip-row {
    display: flex;
    gap: var(--space-3);
    padding: 2px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .heatmap__tooltip-row:last-child {
    border-bottom: none;
  }

  .heatmap__tooltip-key {
    color: var(--text-muted);
    min-width: 40px;
  }

  .heatmap__tooltip-val {
    color: var(--accent-green-300);
    word-break: break-all;
  }

  .heatmap__tooltip-text {
    color: var(--text-primary);
  }

  /* Partial multibyte byte sequence — honesty flag (P1-7), same warning hue
     as the accuracy--approx badge rather than the confident green used for
     normal byte values. */
  .heatmap__tooltip-val--partial {
    color: hsl(40, 90%, 65%);
    cursor: help;
  }
</style>
