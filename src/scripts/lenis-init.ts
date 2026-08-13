/**
 * VKVstudio — Lenis Smooth Scroll Initialization
 *
 * Integrates Lenis with GSAP ticker for smooth scroll behavior.
 * Pattern from design_system.md §7 (Lenis + GSAP Integration).
 *
 * Usage:
 *   import { initLenis, getLenis } from '@scripts/lenis-init';
 *   initLenis();
 */

import Lenis from 'lenis';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

/* ── Register GSAP plugins ─────────────────────────────────── */

gsap.registerPlugin(ScrollTrigger);

/* ── Singleton instance ────────────────────────────────────── */

let lenisInstance: Lenis | null = null;

/* ── EaseOutExpo curve ─────────────────────────────────────── */

function easeOutExpo(t: number): number {
  return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

/* ── Initialize Lenis ──────────────────────────────────────── */

export function initLenis(): Lenis | null {
  // Respect reduced motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return null;
  }

  // Prevent double initialization
  if (lenisInstance) return lenisInstance;

  lenisInstance = new Lenis({
    duration: 1.2,
    easing: easeOutExpo,
    touchMultiplier: 1.5,
  });

  // Expose on window for Svelte islands (they can't import module singletons)
  (window as Window & { lenisInstance?: Lenis }).lenisInstance = lenisInstance;

  // Pipe Lenis scroll events to GSAP ScrollTrigger
  lenisInstance.on('scroll', ScrollTrigger.update);

  // Connect Lenis to GSAP ticker (runs on every frame)
  gsap.ticker.add((time: number) => {
    lenisInstance?.raf(time * 1000);
  });

  // Disable GSAP's lag smoothing for buttery scroll
  gsap.ticker.lagSmoothing(0);

  return lenisInstance;
}

/* ── Get current Lenis instance ────────────────────────────── */

export function getLenis(): Lenis | null {
  return lenisInstance;
}

/* ── Destroy Lenis ─────────────────────────────────────────── */

export function destroyLenis(): void {
  if (lenisInstance) {
    lenisInstance.destroy();
    lenisInstance = null;
  }
}
