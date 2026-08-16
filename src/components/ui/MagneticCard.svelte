<script lang="ts">
  /**
   * MagneticCard — Svelte 5 island for magnetic hover effect on Lab cards
   *
   * Wraps content in a div that follows cursor proximity with easing.
   * Uses runes mode (Svelte 5).
   */

  interface Props {
    strength?: number;
    children?: import('svelte').Snippet;
  }

  let { strength = 0.25, children }: Props = $props();

  let cardEl: HTMLDivElement | undefined = $state();
  let isHovered = $state(false);

  function onMouseMove(e: MouseEvent): void {
    // Respect prefers-reduced-motion (CLAUDE.md house rule): no cursor-follow motion.
    if (!cardEl || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const rect = cardEl.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    cardEl.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
  }

  function onMouseLeave(): void {
    if (!cardEl) return;
    isHovered = false;
    cardEl.style.transform = 'translate(0, 0)';
  }

  function onMouseEnter(): void {
    isHovered = true;
  }
</script>

<div
  class="magnetic-wrapper"
  bind:this={cardEl}
  onmousemove={onMouseMove}
  onmouseleave={onMouseLeave}
  onmouseenter={onMouseEnter}
  class:is-hovered={isHovered}
>
  {@render children?.()}
</div>

<style>
  .magnetic-wrapper {
    display: block;
    height: 100%;
    transition: transform 400ms cubic-bezier(0.34, 1.56, 0.64, 1);
    will-change: transform;
  }

  /* When hovered, respond faster for a natural feel */
  .magnetic-wrapper.is-hovered {
    transition: transform 150ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  @media (prefers-reduced-motion: reduce) {
    .magnetic-wrapper,
    .magnetic-wrapper.is-hovered {
      transition: none;
    }
  }
</style>
