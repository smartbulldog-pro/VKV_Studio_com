<script lang="ts">
  /**
   * SynapseBrain — Fixed circle with scroll-driven video scrub.
   *
   * Always 80×80 circle. Video fills by height (cover fit via canvas).
   * Opacity changes with scroll. At 70%+ becomes clickable.
   */

  import { initBrainMorph } from '@scripts/brain-morph';
  import type { BrainMorphInstance } from '@scripts/brain-morph';
  import { t, type Lang } from '@/i18n/utils';

  // ─── Props ────────────────────────────────────────────────────────────────────
  interface Props {
    /** Optional click callback — when provided, fires instead of tooltip toggle */
    onActivate?: () => void;
    /** Site language for the a11y label / tooltip. Defaults to EN if omitted. */
    lang?: Lang;
  }
  const { onActivate, lang = 'en' }: Props = $props();

  let containerEl: HTMLElement | undefined = $state();
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let videoEl: HTMLVideoElement | undefined = $state();
  let onlineDocEl: HTMLElement | undefined = $state();

  let tooltipVisible: boolean = $state(false);
  let isReady: boolean = $state(false);

  const ariaExpanded = $derived(tooltipVisible ? 'true' : 'false');

  let engine: BrainMorphInstance | null = null;

  $effect(() => {
    if (!containerEl || !canvasEl || !videoEl || !onlineDocEl) return;

    const observer = new MutationObserver(() => {
      isReady = containerEl?.classList.contains('ready') ?? false;
    });
    observer.observe(containerEl, { attributes: true, attributeFilter: ['class'] });

    const heroSection = document.getElementById('hero-section');
    if (!heroSection) return;

    engine = initBrainMorph({
      canvas:      canvasEl,
      video:       videoEl,
      container:   containerEl,
      onlineDoc:   onlineDocEl,
      heroSection: heroSection,
    });

    return () => {
      engine?.destroy();
      engine = null;
      observer.disconnect();
    };
  });

  function handleClick(): void {
    if (!isReady) return;
    // If parent provided onActivate callback (SynapseApp), delegate to it
    if (onActivate) {
      onActivate();
      return;
    }
    // Standalone fallback: toggle tooltip
    tooltipVisible = !tooltipVisible;
  }

  function handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
    if (event.key === 'Escape') {
      tooltipVisible = false;
    }
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  bind:this={containerEl}
  class="synapse-container"
  role="button"
  tabindex={isReady ? 0 : -1}
  aria-label={isReady ? t(lang, 'synapse.brain.open') : t(lang, 'synapse.brain.loading')}
  aria-expanded={ariaExpanded}
  aria-haspopup="dialog"
  onclick={handleClick}
  onkeydown={handleKeydown}
>
  <picture>
    <source srcset="/neural-brain.avif" type="image/avif" />
    <source srcset="/neural-brain.webp" type="image/webp" />
    <img
      src="/neural-brain.png"
      alt={t(lang, 'synapse.brain.alt')}
      class="synapse-fallback"
      aria-hidden="true"
      width="80"
      height="80"
    />
  </picture>

  <!-- NO src attribute in the static markup: on mobile / prefers-reduced-motion,
       brain-morph.ts never assigns one, so the 2.1MB file is never
       requested, mirroring HeroCanvas's hero-scroll.mp4 gating. Desktop
       assigns data-src to src before starting the scrub loop. -->
  <video
    bind:this={videoEl}
    class="synapse-video-source"
    data-src="/brain-morph.mp4"
    data-poster="/neural-brain.webp"
    preload="none"
    muted
    playsinline
    aria-hidden="true"
  ></video>

  <canvas
    bind:this={canvasEl}
    class="synapse-canvas"
    aria-hidden="true"
  ></canvas>

  <span
    bind:this={onlineDocEl}
    class="synapse-online"
    aria-label="Synapse online"
    aria-hidden="true"
  ></span>
</div>

{#if tooltipVisible}
  <div
    class="synapse-tooltip"
    role="dialog"
    aria-label={t(lang, 'synapse.brain.statusLabel')}
    aria-modal="false"
  >
    <p class="synapse-tooltip__text">{t(lang, 'synapse.brain.tooltipTitle')}</p>
    <p class="synapse-tooltip__sub">{t(lang, 'synapse.brain.tooltipSub')}</p>
  </div>
{/if}

<style>
  /* `<picture>` is inline by default — make it a transparent wrapper so the
     absolutely-positioned fallback `<img>` inside lays out exactly as before. */
  picture {
    display: contents;
  }

  /* ── Container: ALWAYS a circle ────────────────────────────────── */
  .synapse-container {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 90;

    width: 80px;
    height: 80px;
    border-radius: 50%;

    background: var(--bg-void);
    overflow: hidden;
    pointer-events: none;
    opacity: 0.6;
    cursor: default;

    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-md);

    transition:
      box-shadow var(--duration-normal) var(--ease-out),
      border-color var(--duration-normal) var(--ease-out);
  }

  /* Step clear of the cookie sheet while it is on screen. Below 640px that
     banner is a full-width sheet flush with the bottom edge and it wins on
     z-index (500 against this element's 90), so on a first visit the entry
     point into the assistant is buried under the consent prompt.
     --consent-sheet-height is published by CookieConsent.svelte only while the
     sheet is visible; the moment it is dismissed the property disappears and
     this collapses back to the plain 24px. */
  @media (max-width: 640px) {
    .synapse-container {
      bottom: calc(24px + var(--consent-sheet-height, 0px));
    }
  }

  @media (max-width: 640px) and (prefers-reduced-motion: no-preference) {
    .synapse-container {
      transition:
        box-shadow var(--duration-normal) var(--ease-out),
        border-color var(--duration-normal) var(--ease-out),
        bottom var(--duration-normal) var(--ease-out);
    }
  }

  /* Ready glow */
  .synapse-container:global(.ready) {
    border-color: var(--border-accent);
    box-shadow:
      var(--shadow-lg),
      var(--glow-green);
  }

  .synapse-container:global(.ready):hover {
    box-shadow:
      var(--shadow-xl),
      0 0 32px var(--accent-glow-strong);
  }

  .synapse-container:focus-visible {
    outline: 2px solid var(--accent-green-300);
    outline-offset: 3px;
  }

  /* ── Canvas ─────────────────────────────────────────────────────── */
  .synapse-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 50%;
  }

  /* ── Hidden video source ────────────────────────────────────────── */
  .synapse-video-source {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }

  /* ── Reduced-motion fallback ───────────────────────────────────── */
  .synapse-fallback {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: none;
    border-radius: 50%;
  }

  @media (prefers-reduced-motion: reduce) {
    .synapse-fallback { display: block; }
    .synapse-canvas   { display: none;  }
    .synapse-container {
      opacity: 1 !important;
    }
  }

  /* ── Online dot ────────────────────────────────────────────────── */
  .synapse-online {
    position: absolute;
    bottom: 6px;
    right: 6px;
    width: 6px;
    height: 6px;
    border-radius: var(--radius-full);
    background: var(--color-success);
    box-shadow: 0 0 6px var(--color-success);
    visibility: hidden;
    opacity: 0;
  }

  /* ── Tooltip ───────────────────────────────────────────────────── */
  .synapse-tooltip {
    position: fixed;
    bottom: calc(24px + 88px);
    right: 24px;
    z-index: 91;

    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);

    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    box-shadow: var(--glass-shadow);

    max-width: 200px;
    animation: tooltip-in var(--duration-slow) var(--ease-out) both;
  }

  .synapse-tooltip__text {
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    margin: 0 0 var(--space-1);
    line-height: var(--leading-snug);
  }

  .synapse-tooltip__sub {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-muted);
    margin: 0;
    letter-spacing: var(--tracking-wide);
  }

  @keyframes tooltip-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Mobile ────────────────────────────────────────────────────── */
  @media (max-width: 767px) {
    .synapse-container {
      left: auto !important;
      right: 16px;
      bottom: 16px;
      transform: none !important;
      width: 64px;
      height: 64px;
    }

    .synapse-tooltip {
      right: 16px;
      bottom: calc(16px + 72px);
    }

    /* brain-morph.mp4 is never fetched on mobile (see brain-morph.ts) —
       show the static neural-brain fallback image instead of a blank
       canvas while the hero is in view. */
    .synapse-fallback { display: block; }
    .synapse-canvas   { display: none;  }

    /* Once "ready" (About section visible), the canvas draws the
       synapse-text ready image — a static asset independent of the
       video — so swap back to match desktop's ready-state look. */
    .synapse-container:global(.ready) .synapse-fallback { display: none;  }
    .synapse-container:global(.ready) .synapse-canvas   { display: block; }
  }

  @media (prefers-reduced-motion: reduce) {
    .synapse-tooltip { animation: none; }
  }
</style>
