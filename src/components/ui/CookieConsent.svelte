<script lang="ts">
  /**
   * CookieConsent.svelte
   * ─────────────────────────────────────────────────────────────────────────────
   * Honest privacy/local-storage notice — not a granular consent manager,
   * because this site currently has nothing non-essential to opt out of:
   * it sets NO cookies at all (language comes from the URL path, not a
   * cookie), uses localStorage/IndexedDB only for on-device functionality
   * (dismissing this notice, caching Lab-tool data and model weights, and
   * local Synapse chat history), and there is no analytics/advertising/
   * tracking of any kind.
   *
   * Behavior
   * ────────
   *  • Shows once, on first visit (checked via localStorage, not a cookie —
   *    using a cookie to gate a cookie notice would be a little too ironic).
   *  • Dismissal ("Got it" / Escape key) persists so it never reappears.
   *  • Non-modal: fixed-position card, no backdrop, never blocks content or
   *    shifts layout (out of document flow the whole time).
   *  • Slide/fade-in motion lives entirely behind
   *    `@media (prefers-reduced-motion: no-preference)`; reduced-motion
   *    users just get an opacity fade.
   *
   * Svelte 5 runes used
   * ───────────────────
   *  • $props()  — typed lang prop
   *  • $state()  — mounted (in DOM at all) / visible (animate-in class)
   *  • $effect() — Escape-key listener, scoped to only while visible
   */
  import type { Lang } from '@i18n/utils';
  import { t } from '@i18n/utils';
  import { onMount } from 'svelte';

  interface Props {
    lang: Lang;
  }

  const { lang }: Props = $props();

  const STORAGE_KEY = 'vkv-cookie-consent-v1';
  const EXIT_DURATION_MS = 320;

  /** Whether the banner exists in the DOM at all (false once dismissed). */
  let mounted = $state(false);
  /** Whether the "visible" (animated-in) class is applied. */
  let visible = $state(false);

  // Set once at mount; only gates the JS-side exit-animation delay below —
  // the CSS animation itself is gated separately via the media query.
  let reducedMotion = false;

  function dismiss(): void {
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      // localStorage unavailable (private browsing / disabled storage) —
      // the banner will simply reappear next visit. Not worth failing over.
    }
    visible = false;
    window.setTimeout(
      () => {
        mounted = false;
      },
      reducedMotion ? 0 : EXIT_DURATION_MS
    );
  }

  onMount(() => {
    reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let alreadyConsented = false;
    try {
      alreadyConsented = localStorage.getItem(STORAGE_KEY) === '1';
    } catch {
      alreadyConsented = false;
    }
    if (alreadyConsented) return;

    mounted = true;
    // Next frame, so the initial (hidden) styles have painted before the
    // "visible" class flips in — otherwise the transition has nothing to
    // transition from.
    requestAnimationFrame(() => {
      visible = true;
    });
  });

  // Escape-key dismissal, only wired up while the banner is actually shown.
  $effect(() => {
    if (!visible) return;

    function onKeydown(e: KeyboardEvent): void {
      if (e.key === 'Escape') dismiss();
    }

    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  });

  // Publish this element's real height so the Synapse brain can step above it.
  // Below 640px the banner is a full-width sheet flush with the bottom edge,
  // and the brain is fixed at bottom:24px / 80px tall with z-index 90 against
  // the sheet's 500 — so on a first visit the site's entry point into the
  // assistant sits underneath the consent prompt. The sheet's height depends on
  // its content and therefore on the locale, so it is measured, not guessed.
  // The property only exists while the sheet is on screen; removing it on
  // teardown is what drops the brain back to its normal position.
  let el = $state<HTMLElement | null>(null);

  $effect(() => {
    const node = el;
    if (!visible || !node) return;

    const root = document.documentElement;
    const publish = (): void =>
      root.style.setProperty('--consent-sheet-height', `${node.offsetHeight}px`);
    publish();

    const ro = new ResizeObserver(publish);
    ro.observe(node);
    return () => {
      ro.disconnect();
      root.style.removeProperty('--consent-sheet-height');
    };
  });
</script>

{#if mounted}
  <div
    bind:this={el}
    class="cookie-consent"
    class:cookie-consent--visible={visible}
    role="region"
    aria-label={t(lang, 'cookieConsent.ariaLabel')}
  >
    <p class="cookie-consent__message">
      {t(lang, 'cookieConsent.message')}
      <a href={`/${lang}/privacy/`} class="cookie-consent__link">
        {t(lang, 'cookieConsent.learnMore')}
      </a>
    </p>
    <button type="button" class="cookie-consent__accept" onclick={dismiss}>
      {t(lang, 'cookieConsent.accept')}
    </button>
  </div>
{/if}

<style>
  .cookie-consent {
    position: fixed;
    left: var(--space-4);
    right: var(--space-4);
    bottom: var(--space-4);
    z-index: 500;
    max-width: 440px;
    margin-inline: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-5);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    box-shadow: var(--glass-shadow);
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--duration-slow) var(--ease-out);
  }

  .cookie-consent--visible {
    opacity: 1;
    pointer-events: auto;
  }

  /* Slide-in transform lives entirely behind the reduced-motion gate —
     users who don't want motion get a plain opacity fade instead. */
  @media (prefers-reduced-motion: no-preference) {
    .cookie-consent {
      transform: translateY(24px);
      transition:
        opacity var(--duration-slow) var(--ease-out),
        transform var(--duration-slow) var(--ease-out);
    }

    .cookie-consent--visible {
      transform: translateY(0);
    }
  }

  .cookie-consent__message {
    font-size: var(--text-sm);
    line-height: var(--leading-relaxed);
    color: var(--text-secondary);
  }

  .cookie-consent__link {
    display: inline-block;
    margin-left: var(--space-1);
    color: var(--accent-green-300);
    text-decoration: underline;
    text-underline-offset: 2px;
    white-space: nowrap;
    transition: color var(--duration-normal) var(--ease-out);
  }

  .cookie-consent__link:hover {
    color: var(--accent-green-200);
  }

  .cookie-consent__link:focus-visible {
    outline: 2px solid var(--accent-green-300);
    outline-offset: 2px;
    border-radius: var(--radius-sm);
  }

  .cookie-consent__accept {
    align-self: flex-end;
    min-height: 44px;
    padding: var(--space-2) var(--space-6);
    background: var(--gradient-accent);
    color: var(--bg-void);
    font-family: var(--font-sans);
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: var(--transition-all);
  }

  .cookie-consent__accept:hover {
    box-shadow: var(--glow-green);
  }

  .cookie-consent__accept:focus-visible {
    outline: 2px solid var(--accent-green-300);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: no-preference) {
    .cookie-consent__accept:hover {
      transform: translateY(-1px);
    }
  }

  /* ── Mobile: bottom sheet ──────────────────────────────────── */
  @media (max-width: 640px) {
    .cookie-consent {
      left: 0;
      right: 0;
      bottom: 0;
      max-width: none;
      border-radius: var(--radius-lg) var(--radius-lg) 0 0;
      padding: var(--space-5) var(--space-5) calc(var(--space-5) + env(safe-area-inset-bottom, 0px));
    }

    .cookie-consent__accept {
      align-self: stretch;
    }
  }

  @media (max-width: 640px) and (prefers-reduced-motion: no-preference) {
    .cookie-consent {
      transform: translateY(100%);
    }

    .cookie-consent--visible {
      transform: translateY(0);
    }
  }
</style>
