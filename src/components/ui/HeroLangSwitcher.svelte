<script lang="ts">
  /**
   * HeroLangSwitcher — Binary Star language switcher for the hero section.
   * Shows orbiting RU/EN stars in the bottom-left corner.
   * Visible only while the hero title is in the viewport.
   * No branches — just the eclipsing binary animation.
   */
  import { getAltLang, t } from '@i18n/utils';

  interface Props {
    lang: 'en' | 'ru';
  }

  let { lang }: Props = $props();

  const altLang = getAltLang(lang);

  /* ── State ─────────────────────────────────────────── */
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let wrapperEl: HTMLElement | undefined = $state();
  let animFrameId: number | null = null;
  let frameCount = 0;
  let orbitAngle = 0;
  let orbitSpeed = 0.008;
  let isHov = false;
  let flashActive = false;
  let flashStart = 0;
  let visible = $state(true);

  const ORBIT_R = 18;

  /* ── Sync visibility with hero overlay opacity ────────── */
  $effect(() => {
    const overlayEl = document.getElementById('hero-overlay');
    if (!overlayEl) return;

    // Check overlay opacity (set by scroll-engine.ts inline style)
    function checkOpacity() {
      const op = parseFloat(overlayEl!.style.opacity || '1');
      visible = op > 0.3;
    }
    checkOpacity();

    // Watch for inline style changes from scroll engine
    const mo = new MutationObserver(checkOpacity);
    mo.observe(overlayEl, { attributes: true, attributeFilter: ['style'] });
    return () => mo.disconnect();
  });

  /* ── Canvas setup & animation ─────────────────────── */
  $effect(() => {
    if (!canvasEl || !wrapperEl) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    function resize() {
      if (!canvasEl || !wrapperEl) return;
      const r = wrapperEl.getBoundingClientRect();
      canvasEl.width = r.width * dpr;
      canvasEl.height = r.height * dpr;
      canvasEl.style.width = `${r.width}px`;
      canvasEl.style.height = `${r.height}px`;
    }
    resize();
    window.addEventListener('resize', resize);

    const ctx = canvasEl.getContext('2d', { alpha: true });
    if (!ctx) return;

    function animate() {
      if (!canvasEl || !ctx) return;
      const w = canvasEl.width / dpr;
      const h = canvasEl.height / dpr;

      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      ctx.save();
      ctx.scale(dpr, dpr);
      frameCount++;

      const lcx = w / 2;
      const lcy = 42; // fixed center — room for stars above and label below

      // Smooth speed lerp
      const targetSpeed = isHov ? 0.022 : 0.008;
      orbitSpeed += (targetSpeed - orbitSpeed) * 0.04;
      orbitAngle += orbitSpeed;

      // Active star (small, bright — current lang)
      const actX = lcx + Math.cos(orbitAngle) * ORBIT_R;
      const actY = lcy + Math.sin(orbitAngle) * ORBIT_R;

      // Companion star (large, dim — alt lang)
      const compX = lcx + Math.cos(orbitAngle + Math.PI) * ORBIT_R;
      const compY = lcy + Math.sin(orbitAngle + Math.PI) * ORBIT_R;

      const breathe = Math.sin(frameCount * 0.03) * 0.08;
      const compR = (isHov ? 13 : 11) * (1 + breathe);

      // --- Dashed orbital line ---
      ctx.save();
      ctx.beginPath();
      ctx.arc(lcx, lcy, ORBIT_R, 0, Math.PI * 2);
      ctx.setLineDash([4, 6]);
      ctx.lineDashOffset = -frameCount * 0.5;
      ctx.strokeStyle = `hsla(185, 50%, 55%, ${isHov ? 0.20 : 0.10})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // --- Comet trail ---
      for (let i = 1; i <= 4; i++) {
        const ta = orbitAngle - i * 0.18;
        const tx = lcx + Math.cos(ta) * ORBIT_R;
        const ty = lcy + Math.sin(ta) * ORBIT_R;
        const tr = 3.5 - i * 0.6;
        const talpha = (0.35 - i * 0.07) * (isHov ? 1.3 : 1);
        if (tr > 0.5) {
          ctx.beginPath();
          ctx.arc(tx, ty, tr, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(185, 80%, 65%, ${Math.max(0, talpha)})`;
          ctx.fill();
        }
      }

      // --- Active star ---
      const actGlow = ctx.createRadialGradient(actX, actY, 0, actX, actY, 16);
      actGlow.addColorStop(0, `hsla(185, 80%, 65%, ${isHov ? 0.35 : 0.25})`);
      actGlow.addColorStop(0.5, `hsla(185, 80%, 65%, 0.08)`);
      actGlow.addColorStop(1, `hsla(185, 80%, 65%, 0)`);
      ctx.beginPath();
      ctx.arc(actX, actY, 16, 0, Math.PI * 2);
      ctx.fillStyle = actGlow;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(actX, actY, 5, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(185, 85%, 70%, 0.92)`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(actX, actY, 2, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(185, 90%, 88%, 0.9)`;
      ctx.fill();

      // --- Companion star ---
      const compGlow = ctx.createRadialGradient(compX, compY, compR * 0.5, compX, compY, compR + 14);
      compGlow.addColorStop(0, `hsla(210, 60%, 55%, ${isHov ? 0.22 : 0.10})`);
      compGlow.addColorStop(1, `hsla(210, 60%, 55%, 0)`);
      ctx.beginPath();
      ctx.arc(compX, compY, compR + 14, 0, Math.PI * 2);
      ctx.fillStyle = compGlow;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(compX, compY, compR, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(210, 35%, 18%, ${isHov ? 0.75 : 0.55})`;
      ctx.fill();
      ctx.strokeStyle = `hsla(185, 55%, 55%, ${isHov ? 0.50 : 0.20})`;
      ctx.lineWidth = isHov ? 1.5 : 1;
      ctx.stroke();
      // Text inside companion
      ctx.font = `700 ${isHov ? 10 : 9}px 'JetBrains Mono', 'SF Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = `hsla(185, 50%, 72%, ${isHov ? 0.95 : 0.65})`;
      ctx.fillText(altLang.toUpperCase(), compX, compY + 0.5);

      // Pulse ring on hover
      if (isHov) {
        const pulse = Math.sin(frameCount * 0.06) * 0.12 + 0.22;
        ctx.beginPath();
        ctx.arc(compX, compY, compR + 8, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(185, 70%, 60%, ${pulse})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // --- Hover label (below) ---
      if (isHov) {
        ctx.font = `500 13px 'JetBrains Mono', 'SF Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const labelText = `→ ${altLang.toUpperCase()}`;
        ctx.shadowColor = `hsla(185, 70%, 60%, 0.5)`;
        ctx.shadowBlur = 14;
        ctx.fillStyle = `hsla(0, 0%, 100%, 0.92)`;
        ctx.fillText(labelText, lcx, lcy + ORBIT_R + 16);
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }

      // --- Flash on click ---
      if (flashActive) {
        const elapsed = frameCount - flashStart;
        const duration = 24;
        if (elapsed < duration) {
          const t = elapsed / duration;
          const eased = 1 - (1 - t) ** 3;
          const flashR = eased * 65;
          const flashAlpha = 1 - eased;
          ctx.beginPath();
          ctx.arc(compX, compY, flashR, 0, Math.PI * 2);
          ctx.strokeStyle = `hsla(185, 90%, 80%, ${flashAlpha * 0.7})`;
          ctx.lineWidth = 3 * (1 - t);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(compX, compY, compR * (1 - t * 0.3), 0, Math.PI * 2);
          ctx.fillStyle = `hsla(185, 90%, 85%, ${flashAlpha * 0.8})`;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(compX, compY, 4 + 16 * (1 - t), 0, Math.PI * 2);
          ctx.fillStyle = `hsla(0, 0%, 100%, ${flashAlpha})`;
          ctx.fill();
        }
      }

      ctx.restore();
      animFrameId = requestAnimationFrame(animate);
    }

    animFrameId = requestAnimationFrame(animate);

    return () => {
      if (animFrameId !== null) cancelAnimationFrame(animFrameId);
      window.removeEventListener('resize', resize);
    };
  });

  /* ── Interaction ───────────────────────────────────── */
  function onMove(e: MouseEvent) {
    if (!wrapperEl) return;
    const rect = wrapperEl.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dist = Math.sqrt((e.clientX - cx) ** 2 + (e.clientY - cy) ** 2);
    isHov = dist < 50;
  }

  function onClick() {
    if (!isHov || flashActive) return;
    flashActive = true;
    flashStart = frameCount;
    setTimeout(() => {
      window.location.href = `/${altLang}/`;
    }, 300);
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="hero-lang"
  class:is-visible={visible}
  bind:this={wrapperEl}
  onmousemove={onMove}
  onmouseleave={() => { isHov = false; }}
  onclick={onClick}
  role="button"
  tabindex="0"
  aria-label={t(lang, 'nav.switchLanguageTo').replace('{lang}', altLang.toUpperCase())}
>
  <canvas
    bind:this={canvasEl}
    class="hero-lang__canvas"
    aria-hidden="true"
  ></canvas>
</div>

<style>
  .hero-lang {
    position: fixed;
    bottom: 3rem;
    left: 2rem;
    width: 80px;
    height: 110px;
    z-index: 20;
    cursor: pointer;
    opacity: 0;
    transform: translateY(12px);
    transition: opacity 0.6s ease-out, transform 0.6s ease-out;
    pointer-events: none;
  }

  .hero-lang.is-visible {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
  }

  .hero-lang__canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  /* Hide when constellation nav is open */
  :global(.constellation-open) .hero-lang {
    opacity: 0 !important;
    pointer-events: none !important;
  }

  @media (max-width: 768px) {
    .hero-lang {
      bottom: 1.5rem;
      left: 1rem;
      width: 64px;
      height: 64px;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .hero-lang {
      transition: none;
    }
  }
</style>
